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

app = FastAPI(title="Trợ Lý KHO Sapo Super Intelligent Engine", version="145.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SHEET_ID = os.getenv("SHEET_ID", "1ZMq0mTiQTDiP92UPaOIv39Q17WJXDiuvrcyYwfs7_Ag").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
SALE_SECRET_KEY = os.getenv("SALE_SECRET_KEY", "sapo2026").strip()

ACTIVE_GROQ_MODEL = None
ACTIVE_GEMINI_MODEL = None

RAM_CACHE_SHEETS = {}
RAM_CACHE_ATTACHMENTS = {}

TABS_PUBLIC = [
    "1_THIET_BI_VA_LOI", 
    "2_HUONG_DAN_CAI_DAT", 
    "3_CHINH_SACH_SAPO", 
    "NHAN_DIEN_THIET_BI"
]
TAB_PRIVATE = "4_DU_LIEU_NOI_BO"
ALL_TABS = TABS_PUBLIC + [TAB_PRIVATE]

HTTP_CLIENT: httpx.AsyncClient = None

# Từ điển đồng nghĩa kỹ thuật thiết bị Sapo
SYNONYMS_DICT = {
    "kẹt dao": ["không cắt giấy", "lỗi cắt giấy", "kẹt dao", "hư dao cắt", "cutter"],
    "khổ giấy": ["kích thước giấy", "khổ tem", "khổ giấy in", "paper size", "kích thước tem"],
    "điện thoại": ["xtest", "app xtest", "in qua lan", "đổi ip", "android", "ios", "máy tính bảng"],
    "máy tính": ["driver", "windows", "mac", "pc", "laptop", "cài driver"],
    "in ra giấy trắng": ["không ra mực", "trắng tinh", "mờ mực", "ngược giấy"]
}

@app.on_event("startup")
async def startup_event():
    global HTTP_CLIENT
    HTTP_CLIENT = httpx.AsyncClient(
        timeout=httpx.Timeout(8.0, read=10.0),
        limits=httpx.Limits(max_keepalive_connections=30, max_connections=150)
    )
    await reload_all_knowledge_base()
    await discover_active_groq_model()
    await discover_active_gemini_model()

@app.on_event("shutdown")
async def shutdown_event():
    if HTTP_CLIENT:
        await HTTP_CLIENT.aclose()

async def discover_active_groq_model():
    """ TỰ ĐỘNG DÒ MODEL GROQ TỐI ƯU NHẤT """
    global ACTIVE_GROQ_MODEL
    if not GROQ_API_KEY:
        return

    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    try:
        res = await HTTP_CLIENT.get(url, headers=headers, timeout=4.0)
        if res.status_code == 200:
            models_data = res.json().get("data", [])
            model_ids = [m["id"] for m in models_data]
            preferred_order = [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "llama3-8b-8192"
            ]
            for pref in preferred_order:
                if pref in model_ids:
                    ACTIVE_GROQ_MODEL = pref
                    return
            if model_ids:
                ACTIVE_GROQ_MODEL = model_ids[0]
                return
    except Exception: pass
    ACTIVE_GROQ_MODEL = "llama-3.1-8b-instant"

async def discover_active_gemini_model():
    """ TỰ ĐỘNG DÒ MODEL GEMINI SỐNG NHẤT TỪ GOOGLE API """
    global ACTIVE_GEMINI_MODEL
    if not GEMINI_API_KEY:
        return

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    try:
        res = await HTTP_CLIENT.get(url, timeout=4.0)
        if res.status_code == 200:
            models_data = res.json().get("models", [])
            valid_models = [
                m["name"].replace("models/", "")
                for m in models_data
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
            preferred_order = [
                "gemini-3.6-flash",
                "gemini-2.5-flash",
                "gemini-2.0-flash",
                "gemini-1.5-flash",
                "gemini-1.5-pro"
            ]
            for pref in preferred_order:
                if pref in valid_models:
                    ACTIVE_GEMINI_MODEL = pref
                    return
            flash_models = [m for m in valid_models if "flash" in m]
            if flash_models:
                ACTIVE_GEMINI_MODEL = flash_models[0]
                return
            if valid_models:
                ACTIVE_GEMINI_MODEL = valid_models[0]
                return
    except Exception: pass
    ACTIVE_GEMINI_MODEL = "gemini-1.5-flash"

async def fetch_single_tab_raw(tab: str):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={tab}"
    try:
        res = await HTTP_CLIENT.get(url, timeout=6.0)
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

async def fetch_attachment_content(url_str: str) -> str:
    """ ĐỌC SÂU NỘI DUNG TẤT CẢ FILE ĐÍNH KÈM: GOOGLE DOCS, PDF, DRIVE & VIDEO """
    if not url_str or url_str in RAM_CACHE_ATTACHMENTS:
        return RAM_CACHE_ATTACHMENTS.get(url_str, "")

    extracted_text = ""
    try:
        # 1. Nếu là Google Doc
        if "docs.google.com/document" in url_str:
            match = re.search(r'/d/([a-zA-Z0-9-_]+)', url_str)
            if match:
                doc_id = match.group(1)
                export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
                res = await HTTP_CLIENT.get(export_url, timeout=5.0)
                if res.status_code == 200:
                    extracted_text = res.text.strip()[:2500]

        # 2. Nếu là File PDF / Google Drive File
        elif "drive.google.com/file" in url_str or "drive.google.com/open" in url_str:
            match = re.search(r'/d/([a-zA-Z0-9-_]+)', url_str)
            if match:
                file_id = match.group(1)
                direct_url = f"https://drive.google.com/uc?id={file_id}&export=download"
                res = await HTTP_CLIENT.get(direct_url, timeout=5.0)
                if res.status_code == 200 and len(res.content) < 3000000:
                    # Trích xuất văn bản thô từ file PDF/Docx
                    raw_str = res.content.decode("utf-8", errors="ignore")
                    clean_str = re.sub(r'[^\w\s\.\:\-\/\(\)]', ' ', raw_str)
                    extracted_text = ' '.join(clean_str.split())[:2000]

        # 3. Nếu là Link Video (YouTube)
        elif "youtube.com" in url_str or "youtu.be" in url_str:
            extracted_text = f"[Video hướng dẫn trực quan]: Xem clip từng bước chi tiết tại link: {url_str}"

    except Exception: pass

    if extracted_text:
        RAM_CACHE_ATTACHMENTS[url_str] = extracted_text
    return extracted_text

async def reload_all_knowledge_base():
    """ TẢI VÀ NẠP TOÀN BỘ TRI THỨC TỪ SHEET + FILE ĐÍNH KÈM VÀO RAM CACHE """
    global RAM_CACHE_SHEETS, RAM_CACHE_ATTACHMENTS
    RAM_CACHE_ATTACHMENTS.clear()

    tasks = [fetch_single_tab_raw(tab) for tab in ALL_TABS]
    results = await asyncio.gather(*tasks)
    RAM_CACHE_SHEETS = {tab: records for tab, records in results}

    # Quét trước toàn bộ link đính kèm để Pre-fetch vào RAM
    attachment_urls = set()
    for records in RAM_CACHE_SHEETS.values():
        for row in records:
            for val in row.values():
                val_str = str(val)
                if "http" in val_str:
                    urls = re.findall(r'https?://[^\s",]+', val_str)
                    attachment_urls.update(urls)

    fetch_tasks = [fetch_attachment_content(u) for u in list(attachment_urls)[:30]]
    await asyncio.gather(*fetch_tasks)
    return {"status": "success", "cached_attachments": len(RAM_CACHE_ATTACHMENTS)}

@app.get("/")
def health_check():
    return {
        "status": "healthy", 
        "active_groq_model": ACTIVE_GROQ_MODEL,
        "active_gemini_model": ACTIVE_GEMINI_MODEL,
        "cached_attachments_count": len(RAM_CACHE_ATTACHMENTS)
    }

@app.get("/reload")
async def reload_data():
    await discover_active_groq_model()
    await discover_active_gemini_model()
    return await reload_all_knowledge_base()

class ChatRequest(BaseModel):
    messages: list
    role: str = "Khach_Hang"

def extract_user_text(event: dict) -> str:
    """ Bóc tách câu hỏi người dùng từ MỌI cấp lồng JSON của Google Chat / GSuite Add-on """
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

    # Bổ sung từ khóa mở rộng từ từ điển đồng nghĩa
    expanded_keywords = [query_lower]
    for key, syns in SYNONYMS_DICT.items():
        if key in query_lower:
            expanded_keywords.extend(syns)

    scored_rows = []
    for tab in accessible_tabs:
        for row in RAM_CACHE_SHEETS.get(tab, []):
            row_text = " ".join(str(v).lower() for v in row.values())
            score = 0
            dev_name = str(row.get("Ten_Thiet_Bi", row.get("Loai_Thiet_Bi", ""))).lower()

            for kw in expanded_keywords:
                if len(kw) >= 2 and kw in dev_name: score += 60
                elif len(kw) >= 2 and kw in row_text: score += 5

            if score > 0:
                scored_rows.append((score, tab, row))

    scored_rows.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored_rows[:3]

    knowledge_text = ""
    for score, tab, row in top_matches:
        knowledge_text += f"\n=== DỮ LIỆU TỪ TAB [{tab}] ===\n"
        for key, value in row.items():
            if value:
                knowledge_text += f"- {key}: {value}\n"
                val_str = str(value)
                # Kiểm tra xem có nội dung file đính kèm được cache trước trong RAM không
                for url, att_text in RAM_CACHE_ATTACHMENTS.items():
                    if url in val_str and att_text:
                        knowledge_text += f"  [NỘI DUNG TÀI LIỆU/PDF ĐÍNH KÈM]: {att_text}\n"

    return knowledge_text

def build_smart_system_prompt(knowledge_context: str) -> str:
    return f"""
    Bạn là Trợ Lý KHO Sapo – Chuyên gia hỗ trợ kỹ thuật thiết bị Sapo CỰC KỲ THÔNG MINH, TINH TẾ, CHÍNH XÁC VÀ TRỰC DIỆN.

    QUY TẮC PHẢN HỒI THÔNG MINH VÀ TINH TẾ (BẮT BUỘC TUÂN THỦ):

    1. **ĐI THẲNG VÀO GIẢI PHÁP / THÔNG SỐ CHÍNH XÁC:**
       - Nếu người dùng hỏi thông số (Ví dụ: "khổ giấy 2 tem", "kích thước tem"): Trả lời NGAY kích thước chi tiết (Rộng x Cao mm và inch).
       - Nếu người dùng hỏi cách cài đặt/sửa lỗi hoặc cung cấp tên model (Ví dụ: "SPL01", "SPR02", "kẹt dao", "in trên điện thoại"): Trả lời NGAY các bước thực hiện chi tiết gạch đầu dòng rõ ràng.
       - **TUYỆT ĐỐI KHÔNG** lặp lại menu hỏi lòng vòng khi người dùng đã đưa ra từ khóa hoặc nhu cầu cụ thể!

    2. **TỰ ĐỘNG TỔNG HỢP VÀ LỌC TRI THỨC:**
       - Lọc ra nội dung quan trọng nhất từ Google Sheet và toàn bộ tài liệu Docs/PDF/Video đính kèm.
       - Trình bày dạng các bước 1-2-3 hoặc gạch đầu dòng trực quan, ngắn gọn, dễ hiểu.

    3. **QUY TẮC ĐỊNH DẠNG TIN NHẮN:**
       - Dùng xưng hô "Em" hoặc "Trợ Lý KHO Sapo", gọi người dùng là "Anh/chị".
       - Đính kèm link hỗ trợ dạng `<URL>` hoặc `[Tên hiển thị](URL)`.
       - Tuyệt đối không vẽ bảng rác làm vỡ giao diện chat.

    DỮ LIỆU KHO TRI THỨC THIẾT BỊ VÀ TÀI LIỆU ĐÍNH KÈM:
    {knowledge_context}
    """

async def call_llm_single(system_instruction: str, user_message: str) -> str:
    """ Gọi AI Groq / Gemini tự động phản hồi tức thì """
    # 1. Gọi Groq Llama 3.3 / Llama 3.1
    if GROQ_API_KEY and ACTIVE_GROQ_MODEL:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": ACTIVE_GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        try:
            res = await HTTP_CLIENT.post(url, headers=headers, json=payload, timeout=6.0)
            if res.status_code == 200:
                data = res.json()
                return data["choices"][0]["message"]["content"]
        except Exception: pass

    # 2. Dự phòng Gemini API
    if GEMINI_API_KEY and ACTIVE_GEMINI_MODEL:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{ACTIVE_GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": [{"text": user_message}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1000}
        }
        try:
            res = await HTTP_CLIENT.post(url, headers=headers, json=payload, timeout=6.0)
            if res.status_code == 200:
                data = res.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception: pass

    return "Dạ em là Trợ Lý KHO Sapo. Anh/chị cần hỗ trợ tra cứu cài đặt hay khắc phục lỗi thiết bị nào ạ?"

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

# ==========================================
# 1. CỔNG WEB VERCEL (/chat) - STREAMING AI
# ==========================================
@app.post("/chat")
async def chat_stream(req: ChatRequest):
    latest_msg = req.messages[-1]["text"] if req.messages else ""
    clean_q = re.sub(r'[^\w\s]', '', latest_msg.lower()).strip()
    quick_greetings = ["chào", "chào bạn", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em"]
    if clean_q in quick_greetings:
        async def greeting_gen():
            yield "Xin chào! Em là **Trợ Lý KHO Sapo**. Anh/chị cần hỗ trợ tra cứu thông số thiết bị hay cài đặt máy in nào ạ?"
        return StreamingResponse(greeting_gen(), media_type="text/plain")

    focused_knowledge = get_high_precision_knowledge(latest_msg, req.role)
    system_instruction = build_smart_system_prompt(focused_knowledge)

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
            "temperature": 0.1,
            "max_tokens": 1000,
            "stream": True
        }

        async def generate_groq():
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
                                        if chunk: yield chunk
                                except Exception: pass
                        return
            except Exception: pass

            async for chunk in generate_gemini_fallback_stream(system_instruction, req.messages):
                yield chunk

        return StreamingResponse(generate_groq(), media_type="text/plain")

    return StreamingResponse(generate_gemini_fallback_stream(system_instruction, req.messages), media_type="text/plain")

async def generate_gemini_fallback_stream(system_instruction: str, messages: list):
    """ Dự phòng Gemini Stream tự động dò model """
    if not GEMINI_API_KEY or not ACTIVE_GEMINI_MODEL:
        yield "Dạ dữ liệu đang được cập nhật, anh/chị thử lại sau giây lát nhé."
        return

    trimmed_messages = messages[-5:] if len(messages) > 5 else messages
    gemini_contents = []
    for m in trimmed_messages:
        role_type = "user" if m["role"] == "user" else "model"
        gemini_contents.append({"role": role_type, "parts": [{"text": m["text"]}]})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{ACTIVE_GEMINI_MODEL}:streamGenerateContent?key={GEMINI_API_KEY}&alt=sse"
    headers = {"Content-Type": "application/json"}
    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": gemini_contents,
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1000}
    }
    try:
        async with HTTP_CLIENT.stream("POST", url, headers=headers, json=payload) as response:
            if response.status_code == 200:
                async for line in response.aiter_lines():
                    if line and line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            data_json = json.loads(data_str)
                            if "candidates" in data_json and len(data_json["candidates"]) > 0:
                                chunk = data_json["candidates"][0]["content"]["parts"][0].get("text", "")
                                if chunk: yield chunk
                        except Exception: pass
    except Exception:
        yield "Dạ em đã nhận thông tin. Anh/chị cần tra cứu thiết bị nào ạ?"

# ==========================================
# 2. CỔNG GOOGLE CHAT BOT (/google-chat)
# ==========================================
@app.post("/google-chat")
async def google_chat_webhook(request: Request):
    try:
        event = await request.json()

        user_message = extract_user_text(event)
        cleaned_message = re.sub(r'<.*?>', '', user_message).replace("@Trợ Lý KHO Sapo", "").strip()

        event_type = event.get("type") or event.get("chat", {}).get("type") or ""

        if event_type == "ADDED_TO_SPACE":
            msg = "👋 Xin chào! Em là Trợ Lý KHO Sapo. Hãy gõ tên thiết bị hoặc câu hỏi để em hỗ trợ ngay 24/7!"
            return JSONResponse(content=wrap_gsuite_addon_response(msg))

        quick_greetings = ["chào", "chào bạn", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em", "chao ban nhe"]
        if not cleaned_message or cleaned_message.lower() in quick_greetings:
            msg = "👋 Xin chào! Em là Trợ Lý KHO Sapo. Anh/chị cần hỗ trợ tra cứu thông số máy in hay cài đặt thiết bị nào ạ?"
            return JSONResponse(content=wrap_gsuite_addon_response(msg))

        focused_knowledge = get_high_precision_knowledge(cleaned_message, role="Sale")
        system_instruction = build_smart_system_prompt(focused_knowledge)

        ai_response = await call_llm_single(system_instruction, cleaned_message)

        return JSONResponse(content=wrap_gsuite_addon_response(ai_response))

    except Exception:
        msg = "Dạ em đã nhận thông tin. Anh/chị cần tra cứu cài đặt hay khắc phục lỗi thiết bị nào ạ?"
        return JSONResponse(content=wrap_gsuite_addon_response(msg))
