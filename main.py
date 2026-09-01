import os
import io
import json
import asyncio
import re
import time
import pandas as pd
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Trợ Lý KHO Sapo Session Cache & Hard Sanitizer Engine", version="440.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# CẤU HÌNH BIẾN MÔI TRƯỜNG & SESSION CACHE
# ------------------------------------------------------------------------------
SHEET_ID = os.getenv("SHEET_ID", "1ZMq0mTiQTDiP92UPaOIv39Q17WJXDiuvrcyYwfs7_Ag").strip()
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "").strip()
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b").strip()
AVAILABLE_CEREBRAS_MODELS = []

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()

RAM_CACHE = {}
# Bộ nhớ tạm lưu tên thiết bị theo ID phòng chat Google Chat để chống mất trí nhớ
GOOGLE_CHAT_SESSION_CACHE = {} 

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
# KHỜI TẠO HTTP CLIENT & DÒ MODEL CEREBRAS
# ------------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global HTTP_CLIENT
    HTTP_CLIENT = httpx.AsyncClient(
        timeout=httpx.Timeout(15.0, read=45.0),
        limits=httpx.Limits(max_keepalive_connections=30, max_connections=100)
    )
    asyncio.create_task(load_sheet_data_async())
    await discover_active_cerebras_models()

@app.on_event("shutdown")
async def shutdown_event():
    if HTTP_CLIENT:
        await HTTP_CLIENT.aclose()

async def discover_active_cerebras_models():
    global CEREBRAS_MODEL, AVAILABLE_CEREBRAS_MODELS
    if not CEREBRAS_API_KEY:
        return

    url = "https://api.cerebras.ai/v1/models"
    headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}"}
    try:
        res = await HTTP_CLIENT.get(url, headers=headers, timeout=6.0)
        if res.status_code == 200:
            models_data = res.json().get("data", [])
            model_ids = [m["id"] for m in models_data]
            AVAILABLE_CEREBRAS_MODELS = model_ids
            if "gpt-oss-120b" in model_ids:
                CEREBRAS_MODEL = "gpt-oss-120b"
            elif "gemma-4-31b" in model_ids:
                CEREBRAS_MODEL = "gemma-4-31b"
            elif model_ids:
                CEREBRAS_MODEL = model_ids[0]
        else:
            CEREBRAS_MODEL = "gpt-oss-120b"
    except Exception:
        CEREBRAS_MODEL = "gpt-oss-120b"

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
        "version": "440.0",
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
# BỘ LỌC CỨNG (HARD SANITIZER) XÓA SẠCH BƯỚC LOCAL PRINTER & CONTROL PANEL
# ------------------------------------------------------------------------------
def clean_thinking_process(text: str) -> str:
    if "Here's a thinking process:" in text:
        parts = text.split("Here's a thinking process:")
        last_part = parts[-1]
        match = re.search(r'(Dạ\s+|Xin chào|Trợ Lý KHO|\*\*|1\.|- )', last_part)
        if match:
            return last_part[match.start():].strip()
    
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'---+', '', text)
    return text.strip()

def sanitize_response_content(text: str) -> str:
    """ Xóa bỏ hoàn toàn mọi câu rác về Add Local Printer / Control Panel """
    raw_clean = clean_thinking_process(text)
    
    forbidden_keywords = [
        "control panel", "add a local printer", "devices and printers", 
        "use an existing port", "add printer or scanner", "thêm máy in (nếu không",
        "thêm máy in thủ công"
    ]
    
    lines = raw_clean.split("\n")
    cleaned_lines = []
    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in forbidden_keywords):
            continue
        cleaned_lines.append(line)
        
    res = "\n".join(cleaned_lines)
    res = re.sub(r'\n{3,}', '\n\n', res)
    return res.strip()

# ------------------------------------------------------------------------------
# TRÍCH XUẤT VÀ LÀM SẠCH DỮ LIỆU
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
                if len(w) >= 2 and w in dev_name: score += 50
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

# ------------------------------------------------------------------------------
# SYSTEM PROMPT BẮT LỢI CẮT NGẮN CÀI DRIVER
# ------------------------------------------------------------------------------
def build_smart_system_prompt(knowledge_context: str) -> str:
    return f"""
# VAI TRÒ & TƯ DUY NGHỆ CƠ BẢN (IDENTITY & EXPERT MINDSET)
Bạn là **Trợ Lý KHO Sapo** – Chuyên gia IT cao cấp phụ trách kỹ thuật phần cứng (máy in đơn hàng, máy in tem, máy quét mã vạch, thiết bị POS...).
- **Phong thái:** Thực chiến, nhạy bén, điềm tĩnh, chuyên nghiệp như một Kỹ thuật viên IT lâu năm.
- **Xưng hô:** Xưng "Em", gọi người dùng là "Anh/chị".

---

# QUY TRÌNH XỬ LÝ THEO KỊCH BẢN (ADAPTIVE WORKFLOWS)

### KỊCH BẢN A: CÂU HỎI TỪ KHÓA CHUNG / CHỈ CÓ TÊN MÁY (Ví dụ: "spr02", "k200l", "xprinter")
- **HÀNH ĐỘNG:** BỎ QUA các chi tiết link trong Kho dữ liệu, TUYỆT ĐỐI KHÔNG xả tài liệu dài dòng hay danh sách lỗi. Chỉ hỏi lại lịch sự để khoanh vùng nhu cầu:
  "Dạ thiết bị **[Tên thiết bị]**, anh/chị đang cần em hỗ trợ mục nào dưới đây ạ?
  1. 💻 **Cài đặt Driver trên Máy tính** (Windows / Mac)
  2. 📱 **Cài đặt in qua Điện thoại / Máy POS** (App XTEST / Kết nối LAN / Đổi IP)
  3. 🛠️ **Khắc phục sự cố** (Không cắt giấy, in ra giấy trắng, nghẽn mạng, báo đèn đỏ...)"

### KỊCH BẢN B: XỬ LÝ SỰ CỐ / BÁO LỖI KỸ THUẬT / CẦN CÀI ĐẶT CỤ THỂ (Ví dụ: "cài driver spr02", "in ra giấy trắng", "1", "driver máy tính nhé")
- **HÀNH ĐỘNG:** 
  1. Trả lời trực diện giải pháp dựa 100% VÀO NỘI DUNG TRONG KHO DỮ LIỆU.
  2. **QUY TRÌNH CÀI DRIVER MÁY TÍNH CHUẨN KHO SAPO (CHỈ GỒM 3 BƯỚC):**
     - **Bước 1 (Tải driver):** Cung cấp link tải tương ứng cho Windows/macOS.
     - **Bước 2 (Cài đặt):** Mở file vừa tải về (`.exe` hoặc `.dmg`), nhấn Next ➔ Install ➔ Finish để hoàn tất.
     - **Bước 3 (In thử):** Kết nối cáp USB và in thử Test Page.
     - **TUYỆT ĐỐI CẤM:** Không hướng dẫn mở Control Panel, Devices and Printers, hay Add a local printer!
  3. Đính kèm ĐẦY ĐỦ link Driver/Tài liệu/Video tương ứng từ Kho dữ liệu bên dưới.

### KỊCH BẢN C: THIẾU THÔNG TIN THIẾT BỊ (ĐIỀN KHUYẾT THÔNG MINH) (Ví dụ: "cài máy in hóa đơn")
- **HÀNH ĐỘNG:** 
  - Nếu trong câu hỏi đã chứa tên máy (VD: SPR02): Sử dụng ngay Kịch bản B, KHÔNG ĐƯỢC hỏi lại model máy in nữa!
  - Nếu thực sự chưa có tên máy: Hỏi khéo: *"Anh/chị cho em xin tên model máy (VD: SPR02, SPL01...) để em gửi chính xác link Driver và video thao tác nhé ạ!"*

---

# LUẬT THÉP BẢO VỆ DỮ LIỆU & CHỐNG LỖI HIỂN THỊ (STRICT GUARDRAILS)

1. **Ngôn ngữ chuẩn 100% Tiếng Việt:** TUYỆT ĐỐI KHÔNG xuất ra dòng suy nghĩ bằng tiếng Anh.
2. **Kiểm soát Link tuyệt đối:** CHỈ CUNG CẤP LINK nếu link đó có mặt 100% chính xác trong KHO DỮ LIỆU.
3. **CẤM DÙNG BẢNG VÀ TIÊU ĐỀ KẺ NGANG:**
   - KHÔNG dùng bảng Markdown (`| ... |`). Liệt kê link dạng gạch đầu dòng emoji:
     - 🔹 **Driver Windows:** `<Link>`
     - 🔹 **Driver macOS:** `<Link>`
   - KHÔNG dùng dấu băm (`#`, `##`, `###`) hay thanh kẻ (`---`).

---

KHO DỮ LIỆU GỐC SAPO:
{knowledge_context}
"""

# ------------------------------------------------------------------------------
# HÀM GỌI GEMINI DỰ PHÒNG
# ------------------------------------------------------------------------------
async def call_gemini_with_retry(system_instruction: str, user_message: str) -> str:
    if not GEMINI_API_KEY:
        return "👋 Dạ em chào anh/chị! Em là **Trợ Lý KHO Sapo**. Anh/chị cần em hỗ trợ cài đặt hay tra cứu thiết bị nào ạ?"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2000}
    }

    try:
        res = await HTTP_CLIENT.post(url, headers=headers, json=payload, timeout=8.0)
        if res.status_code == 200:
            data = res.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            return sanitize_response_content(raw_text)
    except Exception: pass

    return "👋 Dạ em chào anh/chị! Em là **Trợ Lý KHO Sapo**. Anh/chị cần hỗ trợ cài đặt hay tra cứu thiết bị nào ạ?"

async def call_llm_single(system_instruction: str, user_message: str) -> str:
    if CEREBRAS_API_KEY and CEREBRAS_MODEL:
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": CEREBRAS_MODEL,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        try:
            res = await HTTP_CLIENT.post(url, headers=headers, json=payload, timeout=3.5)
            if res.status_code == 200:
                data = res.json()
                return sanitize_response_content(data["choices"][0]["message"]["content"])
        except Exception: pass

    return await call_gemini_with_retry(system_instruction, user_message)

def wrap_gsuite_addon_response(text_message: str) -> dict:
    clean_text = sanitize_response_content(text_message)
    clean_text = re.sub(r'\[(.*?)\]\((https?://.*?)\)', r'\1 (\2)', clean_text)
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

    user_msgs = [m["text"] for m in req.messages if m.get("role") in ["user", "Khach_Hang"]]
    combined_query = " ".join(user_msgs[-3:]) if user_msgs else latest_msg

    focused_knowledge = get_high_precision_knowledge(combined_query, req.role)
    system_instruction = build_smart_system_prompt(focused_knowledge)

    async def generate_response_stream():
        has_yielded = False
        
        if CEREBRAS_API_KEY and CEREBRAS_MODEL:
            messages_payload = [{"role": "system", "content": system_instruction}]
            trimmed = req.messages[-5:] if len(req.messages) > 5 else req.messages
            for m in trimmed:
                role_type = "user" if m["role"] == "user" else "assistant"
                messages_payload.append({"role": role_type, "content": m["text"]})

            url = "https://api.cerebras.ai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": CEREBRAS_MODEL,
                "messages": messages_payload,
                "temperature": 0.1,
                "max_tokens": 2000,
                "stream": True
            }
            try:
                cerebras_timeout = httpx.Timeout(10.0, connect=2.5)
                full_stream_text = ""
                async with HTTP_CLIENT.stream("POST", url, headers=headers, json=payload, timeout=cerebras_timeout) as response:
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
                                        if chunk and not chunk.startswith("<think>"):
                                            full_stream_text += chunk
                                except Exception: pass
                        
                        # Lọc rác bọc lại trước khi nhả ra client
                        clean_output = sanitize_response_content(full_stream_text)
                        if clean_output:
                            has_yielded = True
                            yield clean_output
                            return
            except Exception: pass

        if not has_yielded:
            fallback_ans = await call_gemini_with_retry(system_instruction, combined_query)
            yield fallback_ans

    return StreamingResponse(generate_response_stream(), media_type="text/plain")

# ------------------------------------------------------------------------------
# 2. CỔNG GOOGLE CHAT BOT (/google-chat) - CÓ THÊM BỘ NHỚ PHIÊN SESSION CACHE
# ------------------------------------------------------------------------------
@app.post("/google-chat")
async def google_chat_webhook(request: Request):
    try:
        event = await request.json()
        user_message = extract_user_text(event)
        cleaned_message = re.sub(r'<.*?>', '', user_message).replace("@Trợ Lý KHO Sapo", "").strip()

        # Lấy ID duy nhất của phòng chat hoặc user để lưu Session
        space_id = event.get("space", {}).get("name") or event.get("user", {}).get("name") or "default_space"

        event_type = event.get("type") or event.get("chat", {}).get("type") or ""
        if event_type == "ADDED_TO_SPACE":
            return JSONResponse(content=wrap_gsuite_addon_response("👋 Xin chào! Em là Trợ Lý KHO Sapo. Hãy gõ tên thiết bị hoặc câu hỏi để em hỗ trợ ngay 24/7!"))

        quick_greetings = ["chào", "chào bạn", "chào bjan", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em"]
        if not cleaned_message or any(g == cleaned_message.lower() for g in quick_greetings) or "chào" in cleaned_message.lower():
            return JSONResponse(content=wrap_gsuite_addon_response("👋 Xin chào! Em là Trợ Lý KHO Sapo. Anh/chị cần hỗ trợ tra cứu thông số máy in hay cài đặt thiết bị nào ạ?"))

        # DÒ TÌM VÀ CẬP NHẬT TÊN MÁY VÀO SESSION CACHE
        query_lower = cleaned_message.lower()
        found_device = None
        for tab in TABS_PUBLIC:
            for row in RAM_CACHE.get(tab, []):
                dev_name = str(row.get("Ten_Thiet_Bi", "")).strip()
                if dev_name and dev_name.lower() in query_lower:
                    found_device = dev_name
                    break
            if found_device: break

        if found_device:
            GOOGLE_CHAT_SESSION_CACHE[space_id] = found_device
        
        # NẾU NGƯỜI DÙNG CHỈ GÕ "1", "2", "3" HOẶC "DRIVER MÁY TÍNH", TỰ ĐỘNG GHÉP VỚI TÊN MÁY ĐÃ LƯU TRONG SESSION
        cached_device = GOOGLE_CHAT_SESSION_CACHE.get(space_id)
        final_query = cleaned_message
        if cached_device and (cleaned_message in ["1", "2", "3", "1.", "2.", "3."] or "driver" in query_lower or "máy tính" in query_lower or "điện thoại" in query_lower):
            final_query = f"Thiết bị {cached_device}: {cleaned_message}"

        focused_knowledge = get_high_precision_knowledge(final_query, role="Sale")
        system_instruction = build_smart_system_prompt(focused_knowledge)

        ai_response = await call_llm_single(system_instruction, final_query)
        return JSONResponse(content=wrap_gsuite_addon_response(ai_response))

    except Exception:
        return JSONResponse(content=wrap_gsuite_addon_response("👋 Dạ em chào anh/chị! Em là Trợ Lý KHO Sapo. Anh/chị cần em hỗ trợ cài đặt hay tra cứu lỗi thiết bị nào ạ?"))
