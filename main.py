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

app = FastAPI(title="Trợ Lý KHO Sapo Multi-Tab Intent Engine", version="1600.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# CẤU HÌNH BIẾN MÔI TRƯỜNG & RAM CACHE
# ------------------------------------------------------------------------------
SHEET_ID = os.getenv("SHEET_ID", "1ZMq0mTiQTDiP92UPaOIv39Q17WJXDiuvrcyYwfs7_Ag").strip()
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "").strip()
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gemma-4-31b").strip()
AVAILABLE_CEREBRAS_MODELS = []

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()

RAM_CACHE = {}
GOOGLE_CHAT_HISTORY = {} 

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
# KHỞI TẠO HTTP CLIENT
# ------------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global HTTP_CLIENT
    HTTP_CLIENT = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, read=25.0),
        limits=httpx.Limits(max_keepalive_connections=30, max_connections=100)
    )
    await load_sheet_data_async()
    await discover_active_cerebras_models()

@app.on_event("shutdown")
async def shutdown_event():
    if HTTP_CLIENT:
        await HTTP_CLIENT.aclose()

async def discover_active_cerebras_models():
    global CEREBRAS_MODEL, AVAILABLE_CEREBRAS_MODELS
    if not CEREBRAS_API_KEY: return
    try:
        res = await HTTP_CLIENT.get("https://api.cerebras.ai/v1/models", headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}"}, timeout=5.0)
        if res.status_code == 200:
            model_ids = [m["id"] for m in res.json().get("data", [])]
            AVAILABLE_CEREBRAS_MODELS = model_ids
            gemma_prefs = ["gemma-4-31b", "gemma-4-31b-it", "google/gemma-4-31b-it", "gemma-31b-it", "llama-3.3-70b"]
            for pref in gemma_prefs:
                if pref in model_ids:
                    CEREBRAS_MODEL = pref
                    return
            if model_ids: CEREBRAS_MODEL = model_ids[0]
    except Exception:
        CEREBRAS_MODEL = "gemma-4-31b"

async def fetch_single_tab_raw(tab: str):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={tab}"
    try:
        res = await HTTP_CLIENT.get(url, timeout=6.0)
        if res.status_code == 200 and "text/csv" in res.headers.get("Content-Type", ""):
            df = pd.read_csv(io.BytesIO(res.content)).fillna("")
            records = [{str(k): str(v).strip() for k, v in row.items() if str(v).strip()} for _, row in df.iterrows()]
            return tab, [r for r in records if r]
    except Exception: pass
    return tab, []

async def load_sheet_data_async():
    global RAM_CACHE
    results = await asyncio.gather(*(fetch_single_tab_raw(tab) for tab in ALL_TABS))
    RAM_CACHE = {tab: records for tab, records in results}
    return {"status": "success"}

@app.get("/")
def health_check():
    return {
        "status": "healthy", 
        "version": "1600.0", 
        "engine": "Multi-Tab Intent Engine",
        "active_cerebras_model": CEREBRAS_MODEL,
        "available_cerebras_models": AVAILABLE_CEREBRAS_MODELS,
        "has_cerebras_key": bool(CEREBRAS_API_KEY),
        "has_gemini_key": bool(GEMINI_API_KEY)
    }

@app.get("/reload")
async def reload_data():
    return await load_sheet_data_async()

class ChatRequest(BaseModel):
    messages: list
    role: str = "Khach_Hang"

# ------------------------------------------------------------------------------
# LÀM SẠCH VĂN BẢN (KHỬ MÃ LATEX MŨI TÊN $\rightarrow$)
# ------------------------------------------------------------------------------
def clean_thinking_process(text: str) -> str:
    if "Here's a thinking process:" in text:
        text = text.split("Here's a thinking process:")[-1]
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'---+', '', text)
    
    # Khử sạch mã LaTeX mũi tên dính câu
    text = re.sub(r'\$\\rightarrow\$|\\rightarrow|\$\\Rightarrow\$|\\Rightarrow', '➔', text)
    
    return text.strip()

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
            for v in obj.values():
                res = deep_search(v)
                if res: return res
        return ""

    return deep_search(event)

# ------------------------------------------------------------------------------
# BÓC TÁCH TÊN THIẾT BỊ TỪ LỊCH SỬ CHAT
# ------------------------------------------------------------------------------
def extract_device_from_history(messages: list) -> str:
    device_models = ["spr02", "spr01", "k200l", "k200u", "a868", "hprt", "80fe", "spl01", "xp350b", "g8", "a160m"]
    for m in reversed(messages):
        txt = m.get("text", "").lower()
        for dev in device_models:
            if dev in txt:
                return dev
    return ""

# ------------------------------------------------------------------------------
# TRÍCH XUẤT DỮ LIỆU RAG THÔNG MINH - CHUYỂN TAB CHÍNH XÁC THEO INTENT
# ------------------------------------------------------------------------------
def get_high_precision_knowledge(messages_list: list, role: str) -> str:
    accessible_tabs = ALL_TABS if role == "Sale" else TABS_PUBLIC
    
    latest_msg = messages_list[-1]["text"] if messages_list else ""
    latest_lower = latest_msg.lower()
    
    # 1. Lấy tên thiết bị đã thảo luận trong lịch sử
    detected_dev = extract_device_from_history(messages_list)
    
    # 2. Nhận diện Ý định của câu hỏi MỚI NHẤT
    is_setup_intent = any(k in latest_lower for k in ["cài", "driver", "xtest", "app", "điện thoại", "máy tính", "kết nối", "lan", "ip", "setup"])
    is_error_intent = any(k in latest_lower for k in ["lỗi", "không", "kẹt", "hư", "trắng", "mực", "cắt", "sự cố", "sửa"])
    is_policy_intent = any(k in latest_lower for k in ["bảo hành", "đổi trả", "chính sách", "thu hồi"])

    # 3. Tạo cụm tìm kiếm tập trung câu hỏi mới + Tên thiết bị (Loại bỏ câu hỏi cũ)
    search_phrase = f"{detected_dev} {latest_lower}".strip()
    
    stop_words = {"mình", "có", "bị", "được", "không", "cho", "với", "là", "và", "nhé", "ạ", "cần", "giúp", "tôi", "xin", "lỗi", "thế", "nào", "bao", "nhiêu", "thông", "số", "qua", "đã", "ok", "rồi", "nhưng", "lại", "muốn"}
    words = [w for w in search_phrase.split() if len(w) > 1 and w not in stop_words]

    scored_rows = []
    for tab in accessible_tabs:
        # Thưởng 500 điểm cho Tab khớp trực tiếp với Ý định của câu hỏi mới
        tab_bonus = 0
        if is_setup_intent and tab == "2_HUONG_DAN_CAI_DAT": 
            tab_bonus = 500
        elif is_error_intent and tab == "1_THIET_BI_VA_LOI": 
            tab_bonus = 500
        elif is_policy_intent and tab == "3_CHINH_SACH_SAPO": 
            tab_bonus = 500

        for row in RAM_CACHE.get(tab, []):
            row_text = " ".join(str(v).lower() for v in row.values())
            score = tab_bonus
            
            # Khớp tên thiết bị
            if detected_dev and detected_dev in row_text:
                score += 200
            
            # Khớp từ khóa câu hỏi mới
            for w in words:
                if w in row_text:
                    score += 20
            
            if score > 0:
                scored_rows.append((score, tab, row))

    scored_rows.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored_rows[:3]

    knowledge_text = ""
    for score, tab, row in top_matches:
        knowledge_text += f"\n=== DỮ LIỆU THUỘC TAB [{tab}] ===\n"
        for key, value in row.items():
            if value: 
                knowledge_text += f"- {key}: {value}\n"

    return knowledge_text

# ------------------------------------------------------------------------------
# SYSTEM PROMPT BẢO VỆ ĐỊNH DẠNG & BỘ QUY TẮC CỦA ANH
# ------------------------------------------------------------------------------
def build_smart_system_prompt(knowledge_context: str) -> str:
    return f"""
Bạn là **Trợ Lý KHO Sapo** – Trợ lý AI cao cấp, có tư duy logic sâu sắc và am hiểu kỹ thuật thiết bị Sapo.
Xưng hô: Xưng "Em", gọi "Anh/chị". Phong cách: Lịch sự, chuyên nghiệp, hành văn tự nhiên, rõ ràng.

🎯 QUY TẮC PHẢN HỒI (TUÂN THỦ 100%):

1. TRƯỜNG HỢP 1: CÂU HỎI MẬP MỜ / CHỈ NÓI TÊN THIẾT BỊ / CHỈ NÓI TÊN CHÍNH SÁCH
(Ví dụ: "spr02", "k200l", "chính sách đổi trả", "máy in xprinter", "bảo hành"):
👉 TUYỆT ĐỐI KHÔNG tự đoán mò nhu cầu! KHÔNG xả ngay bài hướng dẫn dài dòng!
👉 BẮT BUỘC hỏi lại 1 câu khoanh vùng nhu cầu:
   - Nếu là THIẾT BỊ (Ví dụ: SPR02, K200L...):
     "Dạ, với thiết bị **[Tên thiết bị]**, anh/chị cần em hỗ trợ mục nào dưới đây ạ?
      1. 💻 Cài đặt trên Máy tính (Windows / Mac)
      2. 📱 Cài đặt trên Điện thoại / App (App XTEST / Kết nối LAN)
      3. 🛠️ Sửa lỗi kỹ thuật / Tra cứu thông số"
   - Nếu là CHÍNH SÁCH / NỘI BỘ (Bảo hành, Thu hồi, Chiết khấu...):
     "Dạ, về **[Chủ đề]**, anh/chị đang cần tra cứu quy định hoặc hướng dẫn cụ thể nào ạ?"

2. TRƯỜNG HỢP 2: CÂU HỎI ĐÃ CÓ Ý ĐỊNH RÕ RÀNG HOẶC HỎI SỬA LỖI / CÀI ĐẶT
(Ví dụ: "cài driver spr02 máy tính", "cài qua app xtest nhé", "spr02 in ra giấy trắng", "chính sách bảo hành 12 tháng"):
👉 Trả lời TRỰC DIỆN từng bước khắc phục/cài đặt có trong Kho dữ liệu bên dưới.

👉 🛑 QUY TẮC ĐÍNH KÈM LINK TỪNG BƯỚC (INLINE STEP-LINK RULE):
   - Nếu ở MỖI BƯỚC CÓ ĐÍNH KÈM LINK ẢNH / LINK VIDEO RIÊNG:
     -> AI BẮT BUỘC phải đặt link ảnh/video đó NGAY DƯỚI BƯỚC TƯƠNG ỨNG.
     -> TUYỆT ĐỐI KHÔNG GỘM NGUYÊN ĐỐNG LINK VỀ CUỐI BÀI!
     -> TUYỆT ĐỐI KHÔNG BỎ SÓT BẤT KỲ LINK NÀO CÓ TRONG DỮ LIỆU GỐC!

🧠 QUY TẮC DUY TRÌ BỐI CẢNH & TRA CỨU DỮ LIỆU (BẮT BUỘC):
   1. GHI NHỚ BỐI CẢNH (Context Memory):
      - Khi người dùng hỏi tiếp trong cuộc trò chuyện (ví dụ: "nhưng in không cắt giấy", "cài qua app xtest nhé"), BẮT BUỘC phải giữ nguyên loại thiết bị/model đang thảo luận ở các câu trước (Ví dụ: SPR02, K200L...). 
      - TUYỆT ĐỐI KHÔNG hỏi lại model máy nếu ở câu trước người dùng hoặc AI đã xác định loại thiết bị đó rồi.

   2. TÁCH TỪ KHÓA CHÍNH VÀ NỘI DUNG MỚI:
      - Khi người dùng chuyển chủ đề từ Sửa lỗi sang Cài đặt (hoặc ngược lại) -> Trích xuất ngay giải pháp tương ứng trong Kho dữ liệu bên dưới để hướng dẫn người dùng đầy đủ các bước.
      - Tuyệt đối không báo "thiếu dữ liệu" khi trong Kho dữ liệu đã cung cấp thông tin.

🛑 LUẬT THÉP ĐỊNH DẠNG:
- TUYỆT ĐỐI CẤM sử dụng mã LaTeX toán học (như $\\rightarrow$, $\\Rightarrow$). Dùng dấu mũi tên "➔" hoặc "->".
- Không tự bịa bước Control Panel hay thao tác phần cứng nếu dữ liệu không có.
- Chỉ cung cấp đường link có thực trong Kho dữ liệu dưới đây.

---

KHO DỮ LIỆU GỐC SAPO:
{knowledge_context}
"""

# ------------------------------------------------------------------------------
# HÀM GỌI LLM CÓ BẢO VỆ CHỐNG TRẢ VỀ CÂU CHÀO KHI LỖI
# ------------------------------------------------------------------------------
async def call_llm_with_history(system_instruction: str, messages_list: list) -> str:
    messages_payload = [{"role": "system", "content": system_instruction}]
    for m in messages_list[-6:]:
        role_type = "user" if m.get("role") in ["user", "Khach_Hang"] else "assistant"
        messages_payload.append({"role": role_type, "content": m.get("text", "")})

    # 1. THỬ GỌI CEREBRAS API
    if CEREBRAS_API_KEY and CEREBRAS_MODEL:
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": CEREBRAS_MODEL,
            "messages": messages_payload,
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 2000
        }
        try:
            res = await HTTP_CLIENT.post(url, headers=headers, json=payload, timeout=10.0)
            if res.status_code == 200:
                data = res.json()
                return clean_thinking_process(data["choices"][0]["message"]["content"])
            else:
                print(f"⚠️ Cerebras lỗi status {res.status_code}: {res.text}")
        except Exception as e:
            print(f"⚠️ Cerebras Exception: {str(e)}")

    # 2. DỰ PHÒNG CHUYỂN SANG GEMINI NẾU CEREBRAS NGHỄN/BẬN
    if GEMINI_API_KEY:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        gemini_contents = []
        for m in messages_list[-6:]:
            role_type = "user" if m.get("role") in ["user", "Khach_Hang"] else "model"
            gemini_contents.append({"role": role_type, "parts": [{"text": m.get("text", "")}]})

        payload = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": gemini_contents,
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2000}
        }
        try:
            res = await HTTP_CLIENT.post(url, headers=headers, json=payload, timeout=8.0)
            if res.status_code == 200:
                data = res.json()
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                return clean_thinking_process(raw_text)
            else:
                print(f"⚠️ Gemini lỗi status {res.status_code}: {res.text}")
        except Exception as e:
            print(f"⚠️ Gemini Exception: {str(e)}")

    return "⚠️ Hệ thống AI hiện đang bận hoặc quá tải lượt truy cập (Lỗi kết nối). Anh/chị vui lòng nhấn gửi lại câu hỏi sau vài giây giúp em nhé! 🙏"

def wrap_gsuite_addon_response(text_message: str) -> dict:
    clean_text = clean_thinking_process(text_message)
    clean_text = re.sub(r'\[(.*?)\]\((https?://.*?)\)', r'\1 (\2)', clean_text)
    clean_text = re.sub(r'\*{2,3}', '*', clean_text)
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
    
    exact_quick_greetings = {"chào", "chào bạn", "chào bjan", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em", "chao ban nhe", "xin chào"}
    if clean_q in exact_quick_greetings:
        async def greeting_gen():
            yield "Xin chào! Em là **Trợ Lý KHO Sapo**. Anh/chị cần hỗ trợ tra cứu thông số thiết bị hay cài đặt máy in nào ạ?"
        return StreamingResponse(greeting_gen(), media_type="text/plain")

    # Gọi RAG lọc dữ liệu chính xác theo Ý định câu hỏi mới + Lịch sử thiết bị
    focused_knowledge = get_high_precision_knowledge(req.messages, req.role)
    system_instruction = build_smart_system_prompt(focused_knowledge)

    async def generate_response_stream():
        ans = await call_llm_with_history(system_instruction, req.messages)
        yield ans

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

        space_id = event.get("space", {}).get("name") or event.get("user", {}).get("name") or "default_space"

        event_type = event.get("type") or event.get("chat", {}).get("type") or ""
        if event_type == "ADDED_TO_SPACE":
            return JSONResponse(content=wrap_gsuite_addon_response("👋 Xin chào! Em là Trợ Lý KHO Sapo. Hãy gõ tên thiết bị hoặc câu hỏi để em hỗ trợ ngay 24/7!"))

        exact_quick_greetings = {"chào", "chào bạn", "chào bjan", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em", "chao ban nhe", "xin chào"}
        clean_user_q = re.sub(r'[^\w\s]', '', cleaned_message.lower()).strip()
        if not cleaned_message or clean_user_q in exact_quick_greetings:
            return JSONResponse(content=wrap_gsuite_addon_response("👋 Xin chào! Em là Trợ Lý KHO Sapo. Anh/chị cần hỗ trợ tra cứu thông số máy in hay cài đặt thiết bị nào ạ?"))

        if space_id not in GOOGLE_CHAT_HISTORY:
            GOOGLE_CHAT_HISTORY[space_id] = []
        
        GOOGLE_CHAT_HISTORY[space_id].append({"role": "user", "text": cleaned_message})
        if len(GOOGLE_CHAT_HISTORY[space_id]) > 10:
            GOOGLE_CHAT_HISTORY[space_id] = GOOGLE_CHAT_HISTORY[space_id][-10:]

        # Gọi RAG lọc dữ liệu chính xác theo Ý định câu hỏi mới + Lịch sử thiết bị
        focused_knowledge = get_high_precision_knowledge(GOOGLE_CHAT_HISTORY[space_id], role="Sale")
        system_instruction = build_smart_system_prompt(focused_knowledge)

        ai_response = await call_llm_with_history(system_instruction, GOOGLE_CHAT_HISTORY[space_id])

        GOOGLE_CHAT_HISTORY[space_id].append({"role": "assistant", "text": ai_response})

        return JSONResponse(content=wrap_gsuite_addon_response(ai_response))

    except Exception:
        return JSONResponse(content=wrap_gsuite_addon_response("⚠️ Hệ thống AI hiện đang bận xử lý. Anh/chị vui lòng nhấn gửi lại câu hỏi sau vài giây giúp em nhé! 🙏"))
