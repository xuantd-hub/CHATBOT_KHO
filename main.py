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

app = FastAPI(title="Trợ Lý KHO Sapo Master Engine", version="143.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SHEET_ID = os.getenv("SHEET_ID", "1ZMq0mTiQTDiP92UPaOIv39Q17WJXDiuvrcyYwfs7_Ag").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
SALE_SECRET_KEY = os.getenv("SALE_SECRET_KEY", "sapo2026").strip()

ACTIVE_GROQ_MODEL = None
RAM_CACHE = {}
DOC_CONTENT_CACHE = {}

TABS_PUBLIC = [
    "1_THIET_BI_VA_LOI", 
    "2_HUONG_DAN_CAI_DAT", 
    "3_CHINH_SACH_SAPO", 
    "NHAN_DIEN_THIET_BI"
]
TAB_PRIVATE = "4_DU_LIEU_NOI_BO"
ALL_TABS = TABS_PUBLIC + [TAB_PRIVATE]

HTTP_CLIENT: httpx.AsyncClient = None

@app.on_event("startup")
async def startup_event():
    global HTTP_CLIENT
    HTTP_CLIENT = httpx.AsyncClient(
        timeout=httpx.Timeout(6.0, read=8.0),
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
    )
    await load_sheet_data_async()
    await discover_active_groq_model()

@app.on_event("shutdown")
async def shutdown_event():
    if HTTP_CLIENT:
        await HTTP_CLIENT.aclose()

async def discover_active_groq_model():
    """ TỰ ĐỘNG DÒ VÀ CHỌN MODEL GROQ THÔNG MINH NHẤT CÓ SẴN """
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
                    print(f"✅ Đã chọn Model Groq tối ưu: {ACTIVE_GROQ_MODEL}")
                    return
            if model_ids:
                ACTIVE_GROQ_MODEL = model_ids[0]
                return
    except Exception as e:
        print(f"⚠️ Lỗi kiểm tra Model Groq: {e}")

    ACTIVE_GROQ_MODEL = "llama-3.1-8b-instant"

async def fetch_single_tab_raw(tab: str):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={tab}"
    try:
        res = await HTTP_CLIENT.get(url, timeout=5.0)
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

async def fetch_google_doc_text(doc_url: str) -> str:
    """ TỰ ĐỘNG TRÍCH XUẤT VĂN BẢN TRONG FILE GOOGLE DOCS ĐÍNH KÈM """
    if not doc_url or "docs.google.com/document" not in doc_url:
        return ""
    
    if doc_url in DOC_CONTENT_CACHE:
        return DOC_CONTENT_CACHE[doc_url]

    try:
        match = re.search(r'/d/([a-zA-Z0-9-_]+)', doc_url)
        if match:
            doc_id = match.group(1)
            export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
            res = await HTTP_CLIENT.get(export_url, timeout=4.0)
            if res.status_code == 200:
                text_content = res.text.strip()
                trimmed_text = text_content[:2000]
                DOC_CONTENT_CACHE[doc_url] = trimmed_text
                return trimmed_text
    except Exception: pass
    return ""

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
        "active_groq_model": ACTIVE_GROQ_MODEL,
        "gemini_model": GEMINI_MODEL
    }

@app.get("/reload")
async def reload_data():
    global DOC_CONTENT_CACHE
    DOC_CONTENT_CACHE.clear()
    return await load_sheet_data_async()

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

async def get_deep_knowledge_context(query: str, role: str) -> str:
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
    top_matches = scored_rows[:2]

    knowledge_text = ""
    for score, tab, row in top_matches:
        knowledge_text += f"\n=== DỮ LIỆU TỪ TAB [{tab}] ===\n"
        for key, value in row.items():
            if value:
                knowledge_text += f"- {key}: {value}\n"
                if "docs.google.com/document" in str(value):
                    doc_text = await fetch_google_doc_text(str(value))
                    if doc_text:
                        knowledge_text += f"  [NỘI DUNG CHI TIẾT FILE DOCS]: {doc_text}\n"

    return knowledge_text

def build_smart_system_prompt(knowledge_context: str) -> str:
    return f"""
    Bạn là Trợ Lý KHO Sapo – Chuyên gia hỗ trợ kỹ thuật thiết bị Sapo CỰC KỲ THÔNG MINH, TRỰC DIỆN và LỊCH SỰ.

    QUY TẮC PHẢN HỒI THÔNG MINH (BẮT BUỘC TUÂN THỦ):

    1. **NẾU CÂU HỎI CÓ Ý ĐỊNH/TÊN THIẾT BỊ RÕ RÀNG (Ví dụ: "cài spr02 trên điện thoại", "máy in kẹt giấy", "khổ giấy 2 tem", "spl01"):**
       - Trả lời NGAY VÀO GIẢI PHÁP/THÔNG SỐ, trình bày ngắn gọn, gạch đầu dòng rõ ràng.
       - Tự động tổng hợp thông tin từ Sheet và file Google Docs đính kèm.
       - **TUYỆT ĐỐI KHÔNG** hiển thị lại menu hỏi lòng vòng khi người dùng đã ghi tên thiết bị hay câu hỏi cụ thể.

    2. **NẾU CÂU HỎI CHỈ LÀ TỪ KHÓA CHUNG CHUNG (Ví dụ: "máy in", "lỗi thiết bị"):**
       - Hỏi lại lịch sự để khoanh vùng nhu cầu:
         "Dạ anh/chị đang cần em hỗ trợ mục nào dưới đây ạ?
         1. 💻 Cài đặt Driver trên Máy tính (Windows / Mac)
         2. 📱 Cài đặt in qua Điện thoại (App XTEST / Đổi IP)
         3. 🛠️ Khắc phục sự cố (Kẹt dao, in ra giấy trắng...)"

    3. **QUY TẮC ĐỊNH DẠNG TIN NHẮN:**
       - Dùng xưng hô "Em" hoặc "Trợ Lý KHO Sapo", gọi "Anh/chị".
       - Đính kèm link chuẩn dạng `<URL>` hoặc `[Tên hiển thị](URL)`.
       - KHÔNG tự vẽ bảng rác.

    DỮ LIỆU TRI THỨC KỸ THUẬT VÀ TÀI LIỆU DOCS:
    {knowledge_context}
    """

async def call_llm_single(system_instruction: str, user_message: str) -> str:
    """ Gọi AI Groq / Gemini dự phòng trả về câu trả lời không streaming """
    # 1. Thử gọi Groq Llama 3.3 / Llama 3.1
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
            "max_tokens": 1000
        }
        try:
            res = await HTTP_CLIENT.post(url, headers=headers, json=payload, timeout=6.0)
            if res.status_code == 200:
                data = res.json()
                return data["choices"][0]["message"]["content"]
        except Exception: pass

    # 2. Dự phòng Gemini API
    if GEMINI_API_KEY:
        gemini_models_to_try = [GEMINI_MODEL, "gemini-1.5-flash", "gemini-1.5-pro"]
        for g_model in gemini_models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{g_model}:generateContent?key={GEMINI_API_KEY}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "systemInstruction": {"parts": [{"text": system_instruction}]},
                "contents": [{"role": "user", "parts": [{"text": user_message}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1000}
            }
            try:
                res = await HTTP_CLIENT.post(url, headers=headers, json=payload, timeout=6.0)
                if res.status_code == 200:
                    data = res.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception: continue

    return "Dạ em là Trợ Lý KHO Sapo. Anh/chị cần hỗ trợ tra cứu cài đặt hay khắc phục lỗi thiết bị nào ạ?"

def wrap_gsuite_addon_response(text_message: str) -> dict:
    """ Đóng gói JSON chuẩn cho Google Chat """
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

    focused_knowledge = await get_deep_knowledge_context(latest_msg, req.role)
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
            "temperature": 0.2,
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
    """ Dự phòng Gemini Stream khi Groq gián đoạn """
    if not GEMINI_API_KEY:
        yield "Dạ dữ liệu đang được cập nhật, anh/chị thử lại sau giây lát nhé."
        return

    trimmed_messages = messages[-5:] if len(messages) > 5 else messages
    gemini_contents = []
    for m in trimmed_messages:
        role_type = "user" if m["role"] == "user" else "model"
        gemini_contents.append({"role": role_type, "parts": [{"text": m["text"]}]})

    gemini_models_to_try = [GEMINI_MODEL, "gemini-1.5-flash", "gemini-1.5-pro"]
    for g_model in gemini_models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{g_model}:streamGenerateContent?key={GEMINI_API_KEY}&alt=sse"
        headers = {"Content-Type": "application/json"}
        payload = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": gemini_contents,
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1000}
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
                    return
        except Exception: continue

# ==========================================
# 2. CỔNG GOOGLE CHAT BOT (/google-chat) - INTELLIGENT AI ENGINE
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

        focused_knowledge = await get_deep_knowledge_context(cleaned_message, role="Sale")
        system_instruction = build_smart_system_prompt(focused_knowledge)

        ai_response = await call_llm_single(system_instruction, cleaned_message)

        return JSONResponse(content=wrap_gsuite_addon_response(ai_response))

    except Exception:
        msg = "Dạ em đã nhận thông tin. Anh/chị cần tra cứu cài đặt hay khắc phục lỗi thiết bị nào ạ?"
        return JSONResponse(content=wrap_gsuite_addon_response(msg))
