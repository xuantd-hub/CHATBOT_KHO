import os
import io
import json
import asyncio
import re
import pandas as pd
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Trợ Lý KHO Sapo Master Engine", version="180.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# CẤU HÌNH BIẾN MÔI TRƯỜNG
# ------------------------------------------------------------------------------
SHEET_ID = os.getenv("SHEET_ID", "1ZMq0mTiQTDiP92UPaOIv39Q17WJXDiuvrcyYwfs7_Ag").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()

ACTIVE_GROQ_MODEL = "llama-3.3-70b-versatile"
RAM_CACHE = {}

TABS_PUBLIC = [
    "1_THIET_BI_VA_LOI", 
    "2_HUONG_DAN_CAI_DAT", 
    "3_CHINH_SACH_SAPO", 
    "NHAN_DIEN_THIET_BI"
]
TAB_PRIVATE = "4_DU_LIEU_NOI_BO"
ALL_TABS = TABS_PUBLIC + [TAB_PRIVATE]

HTTP_CLIENT: httpx.AsyncClient = None

# ------------------------------------------------------------------------------
# KHỞI TẠO VÀ TỰ ĐỘNG CHỌN MODEL GROQ CHUẨN CHAT
# ------------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global HTTP_CLIENT
    HTTP_CLIENT = httpx.AsyncClient(
        timeout=httpx.Timeout(15.0, read=45.0),
        limits=httpx.Limits(max_keepalive_connections=30, max_connections=100)
    )
    asyncio.create_task(load_sheet_data_async())
    await discover_active_groq_model()

@app.on_event("shutdown")
async def shutdown_event():
    if HTTP_CLIENT:
        await HTTP_CLIENT.aclose()

async def discover_active_groq_model():
    global ACTIVE_GROQ_MODEL
    if not GROQ_API_KEY:
        return

    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    try:
        res = await HTTP_CLIENT.get(url, headers=headers, timeout=5.0)
        if res.status_code == 200:
            models_data = res.json().get("data", [])
            model_ids = [m["id"] for m in models_data]
            
            # Ưu tiên các model chat mạnh nhất của Groq (bỏ qua prompt-guard/eval)
            preferred_order = [
                "llama-3.3-70b-versatile",
                "openai/gpt-oss-120b",
                "llama-3.1-8b-instant"
            ]
            for pref in preferred_order:
                if pref in model_ids:
                    ACTIVE_GROQ_MODEL = pref
                    return
    except Exception:
        pass

    ACTIVE_GROQ_MODEL = "llama-3.3-70b-versatile"

async def fetch_single_tab_raw(tab: str):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={tab}"
    try:
        res = await HTTP_CLIENT.get(url, timeout=8.0)
        if res.status_code == 200 and "text/csv" in res.headers.get("Content-Type", ""):
            df = pd.read_csv(io.BytesIO(res.content)).fillna("")
            records = []
            for _, row in df.iterrows():
                row_data = {str(k): str(v).strip() for k, v in row.items() if str(v).strip()}
                if row_data:
                    records.append(row_data)
            return tab, records
    except Exception: pass
    return tab, []

async def load_sheet_data_async():
    global RAM_CACHE
    tasks = [fetch_single_tab_raw(tab) for tab in ALL_TABS]
    results = await asyncio.gather(*tasks)
    RAM_CACHE = {tab: records for tab, records in results}
    return {"status": "success"}

@app.get("/")
def health_check():
    return {
        "status": "healthy", 
        "version": "180.0",
        "active_groq_model": ACTIVE_GROQ_MODEL,
        "backup_gemini_model": GEMINI_MODEL
    }

@app.get("/reload")
async def reload_data():
    return await load_sheet_data_async()

class ChatRequest(BaseModel):
    messages: list
    role: str = "Khach_Hang"

# ------------------------------------------------------------------------------
# HÀM BÓC TÁCH & GHÉP NGỮ CẢNH TÌM KIẾM ĐA LƯỢT
# ------------------------------------------------------------------------------
def extract_user_text(event: dict) -> str:
    if isinstance(event.get("message"), dict):
        msg = event["message"]
        if msg.get("text"): return msg["text"]
        if msg.get("argumentText"): return msg["argumentText"]

    if isinstance(event.get("chat"), dict):
        chat = event["chat"]
        if isinstance(chat.get("messagePayload"), dict) and isinstance(chat["messagePayload"].get("message"), dict):
            m = chat["messagePayload"]["message"]
            if m.get("text"): return m["text"]
            if m.get("argumentText"): return m["argumentText"]
        if isinstance(chat.get("message"), dict):
            m = chat["message"]
            if m.get("text"): return m["text"]
            if m.get("argumentText"): return m["argumentText"]

    def deep_search(obj):
        if isinstance(obj, dict):
            if "argumentText" in obj and isinstance(obj["argumentText"], str) and obj["argumentText"].strip():
                return obj["argumentText"]
            if "text" in obj and isinstance(obj["text"], str) and obj["text"].strip():
                if not obj["text"].startswith("spaces/"):
                    return obj["text"]
            for k, v in obj.items():
                res = deep_search(v)
                if res: return res
        return ""

    return deep_search(event)

def get_high_precision_knowledge(query: str, role: str) -> str:
    accessible_tabs = ALL_TABS if role == "Sale" else TABS_PUBLIC
    query_lower = query.lower()
    stop_words = {"mình", "có", "bị", "được", "không", "cho", "với", "là", "và", "nhé", "ạ", "cần", "giúp", "tôi", "xin", "lỗi", "máy", "thế", "nào", "bao", "nhiêu", "thông", "số", "in", "qua", "đã", "ok"}
    words = [w for w in query_lower.split() if len(w) > 1 and w not in stop_words]
    if not words: words = [query_lower]

    scored_rows = []
    for tab in accessible_tabs:
        for row in RAM_CACHE.get(tab, []):
            row_text = " ".join(str(v).lower() for v in row.values())
            score = 0
            dev_name = str(row.get("Ten_Thiet_Bi", row.get("Loai_Thiet_Bi", ""))).lower()
            for w in words:
                if len(w) >= 3 and w in dev_name: score += 50
                elif w in row_text: score += 3
            if score > 0: scored_rows.append((score, tab, row))

    scored_rows.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored_rows[:3]

    knowledge_text = ""
    for score, tab, row in top_matches:
        knowledge_text += f"\n=== DỮ LIỆU TỪ TAB [{tab}] ===\n"
        for key, value in row.items():
            if value: knowledge_text += f"- {key}: {value}\n"
    return knowledge_text

def clean_thinking_process(text: str) -> str:
    """ Loại bỏ đoạn suy nghĩ Tiếng Anh (Thinking Process) của các model reasoning """
    if "Here's a thinking process:" in text:
        parts = text.split("Here's a thinking process:")
        last_part = parts[-1]
        # Tìm vị trí câu trả lời thực sự bằng Tiếng Việt
        match = re.search(r'(Dạ\s+|Xin chào|Trợ Lý KHO|\*\*|1\.|- )', last_part)
        if match:
            return last_part[match.start():].strip()
    
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()

def build_smart_system_prompt(knowledge_context: str) -> str:
    return f"""
# VAI TRÒ & TƯ DUY NGHỆ CƠ BẢN (IDENTITY & EXPERT MINDSET)

Bạn là **Trợ Lý KHO Sapo** – Chuyên gia IT cao cấp phụ trách kỹ thuật phần cứng (máy in đơn hàng, máy in tem, máy quét mã vạch, thiết bị POS...).
- **Phong thái:** Thực chiến, nhạy bén, điềm tĩnh, chuyên nghiệp như một Kỹ thuật viên IT lâu năm.
- **Xưng hô:** Xưng "Em", gọi người dùng là "Anh/chị".
- **Tư duy cốt lõi (Root Cause Analysis):** Luôn phân tích sự cố theo chiều hướng: **Phần cứng (Điện, dây, giấy) ➡️ Phần mềm (Driver, Khổ giấy) ➡️ Kết nối (IP, LAN, Bluetooth)**. Hướng dẫn người dùng làm bước ĐƠN GIẢN NGHĨA LÀ TRƯỚC, BỚT PHỨC TẠP SAU.

---

# MA TRẬN LIÊN KẾT CHÉO DỮ LIỆU (CROSS-DATA LINKAGE)

Khi nhận câu hỏi, bạn phải tự động kích hoạt bộ lọc liên kết 4 chiều:
1. **Model thiết bị:** (SPR02, SPL01, K200L, Xprinter...) 
2. **Hệ điều hành / Thiết bị điều khiển:** (Windows, Mac, Android, iOS, Máy POS Sapo)
3. **Cổng kết nối:** (USB, LAN, Bluetooth, Wi-Fi)
4. **Mục đích sử dụng:** (In hóa đơn, in tem nhãn, in đơn hàng sàn TMĐT...)

*Ví dụ tư duy:* Nếu người dùng hỏi "In không ra chữ trên máy SPR02 qua điện thoại" ➡️ Tự động liên kết: SPR02 (Máy in nhiệt) + Điện thoại (Kết nối App/IP) + In ra giấy trắng (Lắp ngược cuộn giấy nhiệt HOẶC sai IP) ➡️ Đưa ra chẩn đoán chính xác ngay lập tức.

---

# QUY TRÌNH XỬ LÝ THEO KỊCH BẢN (ADAPTIVE WORKFLOWS)

### KỊCH BẢN A: CÂU HỎI TỪ KHÓA CHUNG / CHỈ CÓ TÊN MÁY
*(Ví dụ: "spr02", "k200l", "xprinter")*
- **HÀNH ĐỘNG:** BỎ QUA các chi tiết link trong Kho dữ liệu, TUYỆT ĐỐI KHÔNG xả tài liệu dài dòng. Chỉ hỏi lại lịch sự để thu hẹp phạm vi hỗ trợ:
  "Dạ thiết bị **[Tên thiết bị]**, anh/chị đang cần em hỗ trợ mục nào dưới đây ạ?
  1. 💻 **Cài đặt Driver trên Máy tính** (Windows / Mac)
  2. 📱 **Cài đặt in qua Điện thoại / Máy POS** (App XTEST / Kết nối LAN / Đổi IP)
  3. 🛠️ **Khắc phục sự cố** (Không cắt giấy, in ra giấy trắng, nghẽn mạng, báo đèn đỏ...)"

### KỊCH BẢN B: XỬ LÝ SỰ CỐ / BÁO LỖI KỸ THUẬT / CẦN CÀI ĐẶT CỤ THỂ
*(Ví dụ: "in ra giấy trắng", "cài spr02 qua điện thoại", "driver spr02")*
- **HÀNH ĐỘNG:** Đưa ra giải pháp trực diện, ngắn gọn theo luồng 3 bước kỹ thuật:
  - **Bước 1: Kiểm tra nhanh phần cứng** (Công tắc, dây cáp, chiều lắp giấy nhiệt).
  - **Bước 2: Cấu hình phần mềm/mạng** (Chỉnh driver, cài đúng IP, chọn đúng khổ giấy).
  - **Bước 3: Hướng dẫn nâng cao / Link cài đặt** (Đính kèm đầy đủ link Driver/Tài liệu tương ứng từ Kho dữ liệu).

### KỊCH BẢN C: THIẾU THÔNG TIN THIẾT BỊ (ĐIỀN KHUYẾT THÔNG MINH)
*(Ví dụ: "cài máy in hóa đơn", "in tem bị chệch")*
- **HÀNH ĐỘNG:** 
  - Nếu trong các tin nhắn trước người dùng ĐÃ NÓI tên thiết bị: Dùng ngay tên thiết bị đó để xử lý theo Kịch bản B, KHÔNG HỎI LAI.
  - Nếu người dùng CHƯA NÓI tên thiết bị: Đưa ngay quy trình xử lý chuẩn IT chung (VD: hướng dẫn vào Control Panel > Devices and Printers) 💬 **ĐỒNG THỜI** kết bài bằng lời hỏi khéo: *"Anh/chị cho em xin tên model máy (VD: SPR02, SPL01...) để em gửi chính xác link Driver và video thao tác nhé ạ!"*

### KỊCH BẢN D: THIẾT BỊ NẰM NGOÀI DANH MỤC
*(Ví dụ: Người dùng hỏi về máy in Canon, Epson... không có trong kho dữ liệu)*
- **HÀNH ĐỘNG:** Trả lời lịch sự: *"Dạ thiết bị này hiện nằm ngoài danh mục thiết bị chuẩn do Sapo cung cấp. Tuy nhiên theo chuẩn kỹ thuật chung, anh/chị có thể kiểm tra [đưa ra 1-2 bước xử lý IT cơ bản]. Nếu cần hỗ trợ thêm, anh/chị vui lòng liên hệ tổng đài Sapo nhé ạ!"*

---

# LUẬT THÉP BẢO VỆ DỮ LIỆU & CHỐNG BỊA ĐẶT (STRICT GUARDRAILS)

1. **Ngôn ngữ chuẩn 100% Tiếng Việt:** TUYỆT ĐỐI KHÔNG để lọt các dòng suy nghĩ bằng tiếng Anh (như "Analyzing prompt...", "Thought process...").
2. **Kiểm soát Link tuyệt đối (Zero Hallucinated URLs):** 
   - CHỈ ĐƯỢC CUNG CẤP LINK nếu link đó có mặt 100% chính xác trong `{knowledge_context}`.
   - Nếu KHÔNG CÓ LINK trong kho dữ liệu ➡️ Tuyệt đối KHÔNG tự bịa link dạng `sapo.vn/...` hay link ngoài. Hãy hướng dẫn bằng lời hoặc bảo: *"Dạ phần này em chưa có sẵn file tải trực tiếp, em hướng dẫn anh/chị thao tác trực tiếp trên máy nhé!"*
3. **Chính xác hotline:** Số tổng đài / hotline kỹ thuật bắt buộc phải trích xuất chuẩn xác từ tài liệu chính sách.
4. **Vận dụng tri thức IT an toàn:** Được phép dùng kiến thức IT chuẩn (thao tác Windows, Mac, IP, Control Panel) để giải thích chi tiết, nhưng các thông số riêng của Sapo phải bám sát Kho dữ liệu.

---

# CHUẨN TRÌNH BÀY DÀNH CHO KỸ THUẬT (FORMATTING RULES)

- **Đường dẫn thao tác rõ ràng:** Dùng dấu `>` để hướng dẫn từng bước trên phần mềm (Ví dụ: **Control Panel** > **View devices and printers** > Click chuột phải chọn **Printer Properties**).
- **Trình bày:** Sử dụng gạch đầu dòng, các từ khóa quan trọng và tên nút bấm phải **in đậm** bằng `**từ khóa**`.
- **Tương tác:** Thêm emoji trực quan (💻, 📱, 🔌, ⚠️, 📌) giúp đoạn văn dễ đọc.
- **CẤM DÙNG BẢNG:** Tuyệt đối KHÔNG xuất bảng Markdown dưới mọi hình thức.

---

# KHO DỮ LIỆU GỐC SAPO
{knowledge_context}
    """

# ------------------------------------------------------------------------------
# HÀM GỌI GEMINI 3.6 FLASH DỰ PHÒNG
# ------------------------------------------------------------------------------
async def call_gemini_with_retry(system_instruction: str, user_message: str) -> str:
    if not GEMINI_API_KEY:
        return "👋 Dạ em là Trợ Lý KHO Sapo. Anh/chị cần hỗ trợ tra cứu thiết bị nào ạ?"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }
    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2000}
    }

    for attempt in range(3):
        try:
            res = await HTTP_CLIENT.post(url, headers=headers, json=payload, timeout=12.0)
            if res.status_code == 200:
                data = res.json()
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                return clean_thinking_process(raw_text)
            elif res.status_code in [503, 429]:
                await asyncio.sleep(1.0)
                continue
        except Exception:
            await asyncio.sleep(1.0)

    return "👋 Dạ em đã nhận thông tin. Anh/chị cần hỗ trợ tra cứu thông số hay cài đặt thiết bị nào ạ?"

async def call_llm_single(system_instruction: str, user_message: str) -> str:
    if GROQ_API_KEY and ACTIVE_GROQ_MODEL:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": ACTIVE_GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.2,
            "max_tokens": 2000
        }
        try:
            res = await HTTP_CLIENT.post(url, headers=headers, json=payload, timeout=12.0)
            if res.status_code == 200:
                data = res.json()
                raw_text = data["choices"][0]["message"]["content"]
                return clean_thinking_process(raw_text)
        except Exception:
            pass

    return await call_gemini_with_retry(system_instruction, user_message)

def wrap_gsuite_addon_response(text_message: str) -> dict:
    clean_text = re.sub(r'\[(.*?)\]\((https?://.*?)\)', r'\1 (\2)', text_message)
    return {
        "hostAppDataAction": {
            "chatDataAction": {
                "createMessageAction": {
                    "message": {
                        "text": clean_text
                    }
                }
            }
        }
    }

# ------------------------------------------------------------------------------
# 1. CỔNG WEB CHAT (/chat)
# ------------------------------------------------------------------------------
@app.post("/chat")
async def chat_stream(req: ChatRequest):
    latest_msg = req.messages[-1]["text"] if req.messages else ""
    clean_q = re.sub(r'[^\w\s]', '', latest_msg.lower()).strip()
    
    quick_greetings = ["chào", "chào bạn", "chào bjan", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em", "chao ban nhe"]
    if clean_q in quick_greetings or "chào" in clean_q or "chao" in clean_q:
        async def greeting_gen():
            yield "Xin chào! Em là **Trợ Lý KHO Sapo**. Anh/chị cần hỗ trợ tra cứu thông số thiết bị hay cài đặt máy in nào ạ?"
        return StreamingResponse(greeting_gen(), media_type="text/plain")

    # ⚡ ĐIỂM SỬA QUAN TRỌNG: Ghép 3 câu nói gần nhất để KHÔNG BỊ MẤT TÊN MÁY (VD: SPR02) ở câu thứ 2
    user_msgs = [m["text"] for m in req.messages if m.get("role") in ["user", "Khach_Hang"]]
    combined_query = " ".join(user_msgs[-3:]) if user_msgs else latest_msg

    focused_knowledge = get_high_precision_knowledge(combined_query, req.role)
    system_instruction = build_smart_system_prompt(focused_knowledge)

    async def generate_response_stream():
        has_yielded = False
        
        # 1. ƯU TIÊN GROQ STREAMING
        if GROQ_API_KEY and ACTIVE_GROQ_MODEL:
            messages_payload = [{"role": "system", "content": system_instruction}]
            trimmed = req.messages[-5:] if len(req.messages) > 5 else req.messages
            for m in trimmed:
                role_type = "user" if m["role"] == "user" else "assistant"
                messages_payload.append({"role": role_type, "content": m["text"]})

            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": ACTIVE_GROQ_MODEL,
                "messages": messages_payload,
                "temperature": 0.2,
                "max_tokens": 2000,
                "stream": True
            }
            try:
                async with HTTP_CLIENT.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code == 200:
                        async for line in response.aiter_lines():
                            if line and line.startswith("data: "):
                                data_str = line[6:].strip()
                                if data_str == "[DONE]": break
                                try:
                                    data_json = json.loads(data_str)
                                    choices = data_json.get("choices", [])
                                    if choices:
                                        chunk = choices[0].get("delta", {}).get("content", "")
                                        if chunk:
                                            has_yielded = True
                                            yield chunk
                                except Exception: pass
                        if has_yielded:
                            return
            except Exception: pass

        # 2. DỰ PHÒNG GEMINI 3.6 FLASH
        if not has_yielded:
            fallback_ans = await call_gemini_with_retry(system_instruction, latest_msg)
            yield fallback_ans

    return StreamingResponse(generate_response_stream(), media_type="text/plain")

# ------------------------------------------------------------------------------
# 2. CỔNG GOOGLE CHAT BOT (/google-chat)
# ------------------------------------------------------------------------------
@app.post("/google-chat")
async def google_chat_webhook(request: Request):
    try:
        event = await request.json()
        user_message = extract_user_text(event)
        cleaned_message = re.sub(r'<.*?>', '', user_message).replace("@Trợ Lý KHO Sapo", "").strip()

        event_type = event.get("type") or event.get("chat", {}).get("type") or ""
        if event_type == "ADDED_TO_SPACE":
            return JSONResponse(content=wrap_gsuite_addon_response("👋 Xin chào! Em là Trợ Lý KHO Sapo. Hãy gõ tên thiết bị hoặc câu hỏi để em hỗ trợ ngay 24/7!"))

        quick_greetings = ["chào", "chào bạn", "chào bjan", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em"]
        if not cleaned_message or cleaned_message.lower() in quick_greetings or "chào" in cleaned_message.lower():
            return JSONResponse(content=wrap_gsuite_addon_response("👋 Xin chào! Em là Trợ Lý KHO Sapo. Anh/chị cần hỗ trợ tra cứu thông số máy in hay cài đặt thiết bị nào ạ?"))

        focused_knowledge = get_high_precision_knowledge(cleaned_message, role="Sale")
        system_instruction = build_smart_system_prompt(focused_knowledge)

        ai_response = await call_llm_single(system_instruction, cleaned_message)
        return JSONResponse(content=wrap_gsuite_addon_response(ai_response))

    except Exception:
        return JSONResponse(content=wrap_gsuite_addon_response("👋 Dạ em đã nhận thông tin. Anh/chị cần tra cứu cài đặt hay khắc phục lỗi thiết bị nào ạ?"))
