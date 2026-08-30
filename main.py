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

app = FastAPI(title="Trợ Lý KHO Smart & UltraFast Engine", version="113.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SHEET_ID = os.getenv("SHEET_ID", "1ZMq0mTiQTDiP92UPaOIv39Q17WJXDiuvrcyYwfs7_Ag").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6Lv4_HzCEz6iLuRChDrw-NGLOO28NYuM37uBe8caeYIZg").strip()
SALE_SECRET_KEY = os.getenv("SALE_SECRET_KEY", "sapo2026").strip()

ACTIVE_GROQ_MODEL = None
RAM_CACHE = {}

TABS_PUBLIC = [
    "1_THIET_BI_VA_LOI", 
    "2_HUONG_DAN_CAI_DAT", 
    "3_CHINH_SACH_SAPO", 
    "NHAN_DIEN_THIET_BI"
]
TAB_PRIVATE = "4_DU_LIEU_NOI_BO"
ALL_TABS = TABS_PUBLIC + [TAB_PRIVATE]

# TỪ ĐIỂN ĐỒNG NGHĨA KỸ THUẬT KHO (TĂNG ĐỘ THÔNG MINH)
SYNONYM_MAP = {
    "giấy": ["tem", "decal", "giấy in", "khổ tem", "cuộn"],
    "mã vạch": ["barcode", "tem", "xprinter", "spl01", "g8", "nhãn"],
    "khổ": ["kích thước", "size", "khổ in", "chiều rộng"],
    "lỗi": ["sự cố", "không in", "kẹt", "báo đỏ", "hỏng", "kêu"],
    "cài": ["driver", "hướng dẫn", "setup", "lắp đặt", "kết nối"]
}

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
                "llama3-8b-8192",
                "mixtral-8x7b-32768"
            ]
            for pref in preferred_order:
                if pref in model_ids:
                    ACTIVE_GROQ_MODEL = pref
                    print(f"✅ [GROQ MODEL]: {ACTIVE_GROQ_MODEL}")
                    return
            if model_ids:
                ACTIVE_GROQ_MODEL = model_ids[0]
    except Exception as e:
        print(f"⚠️ Lỗi check Groq model: {e}")
    ACTIVE_GROQ_MODEL = "llama3-8b-8192"

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
    except Exception as e:
        print(f"⚠️ Cảnh báo tab '{tab}': {e}")
    return tab, []

async def load_sheet_data_async():
    global RAM_CACHE
    tasks = [fetch_single_tab_raw(tab) for tab in ALL_TABS]
    results = await asyncio.gather(*tasks)
    RAM_CACHE = {tab: records for tab, records in results}
    print("✅ [CLOUD RUN] Đã nạp dữ liệu RAM!")
    return {"status": "success"}

@app.get("/")
def health_check():
    return {"status": "healthy", "model": ACTIVE_GROQ_MODEL}

@app.get("/reload")
async def reload_data():
    return await load_sheet_data_async()

class SaleAuthRequest(BaseModel):
    email: str
    passcode: str

@app.post("/verify-sale")
def verify_sale(req: SaleAuthRequest):
    email = req.email.strip().lower()
    passcode = req.passcode.strip()
    if not email.endswith("@sapo.vn"):
        return {"success": False, "message": "Email phải có đuôi @sapo.vn!"}
    if passcode == SALE_SECRET_KEY:
        return {"success": True, "message": "Xác thực Sale thành công!"}
    return {"success": False, "message": "Mật khẩu nội bộ chưa chính xác!"}

class ChatRequest(BaseModel):
    messages: list
    role: str = "Khach_Hang"

def get_smart_focused_knowledge(query: str, role: str) -> str:
    """ Thuật toán RAG Thông Minh: Tự động tra từ điển đồng nghĩa & mở rộng 8 kết quả """
    stop_words = {"mình", "có", "bị", "được", "không", "cho", "với", "là", "và", "nhé", "ạ", "cần", "giúp", "tôi", "xin", "lỗi", "máy", "thế", "nào", "bao", "nhiêu", "thông", "số"}
    raw_words = [w.lower() for w in query.split() if len(w) > 1 and w.lower() not in stop_words]
    
    # Mở rộng từ khóa dựa vào Từ Điển Đồng Nghĩa
    expanded_search_terms = set(raw_words)
    for word in raw_words:
        if word in SYNONYM_MAP:
            expanded_search_terms.update(SYNONYM_MAP[word])

    accessible_tabs = ALL_TABS if role == "Sale" else TABS_PUBLIC
    scored_rows = []

    for tab in accessible_tabs:
        for row in RAM_CACHE.get(tab, []):
            row_text = " ".join(str(v).lower() for v in row.values())
            score = 0
            for term in expanded_search_terms:
                if term in row_text:
                    score += 2
            model_name = str(row.get("Ten_Thiet_Bi", "")).lower()
            if any(term in model_name for term in expanded_search_terms):
                score += 10
            if score > 0:
                scored_rows.append((score, tab, row))

    scored_rows.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored_rows[:8]  # Tăng lên 8 mục để AI có góc nhìn sâu hơn

    if not top_matches:
        top_matches = [(1, "2_HUONG_DAN_CAI_DAT", r) for r in RAM_CACHE.get("2_HUONG_DAN_CAI_DAT", [])[:4]]

    knowledge_text = ""
    for score, tab, row in top_matches:
        knowledge_text += f"\n--- [Nguồn: {tab}] ---\n"
        for key, value in row.items():
            knowledge_text += f"{key}: {value}\n"
    return knowledge_text

# ==========================================
# 1. CỔNG WEB VERCEL (/chat)
# ==========================================
@app.post("/chat")
async def chat_stream(req: ChatRequest):
    latest_msg = req.messages[-1]["text"] if req.messages else ""
    focused_knowledge = get_smart_focused_knowledge(latest_msg, req.role)

    system_instruction = f"""
    Bạn là Trợ Lý KHO Sapo – Chuyên gia tư vấn & kỹ thuật phần cứng Sapo. Thông minh, nhạy bén và chuyên nghiệp.

    NHIỆM VỤ:
    1. Phân tích Kho Tri Thức bên dưới để giải đáp chính xác, tự động kết nối từ khóa đồng nghĩa (VD: "khổ giấy mã vạch" -> "kích thước tem in").
    2. Hướng dẫn chi tiết từng bước, trình bày đẹp mắt bằng Markdown.
    3. Đính kèm ĐẦY ĐỦ link Driver/Video có trong dữ liệu bằng cú pháp `[Tên hiển thị](URL)`.
    4. Xưng danh "Trợ Lý KHO". Dùng `➔` chỉ hướng.

    KHO TRI THỨC KỸ THUẬT:
    {focused_knowledge}
    """

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
            "max_tokens": 1200,
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
            
            async for chunk in generate_gemini_stream(system_instruction, req.messages):
                yield chunk

        return StreamingResponse(generate_groq(), media_type="text/plain", headers={"Cache-Control": "no-cache"})

    return StreamingResponse(generate_gemini_stream(system_instruction, req.messages), media_type="text/plain", headers={"Cache-Control": "no-cache"})

async def generate_gemini_stream(system_instruction: str, messages: list):
    trimmed_messages = messages[-5:] if len(messages) > 5 else messages
    gemini_contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["text"]}]} for m in trimmed_messages]

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:streamGenerateContent?key={GEMINI_API_KEY}&alt=sse"
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
    except Exception as err:
        yield f"❌ Lỗi AI: {str(err)}"

# ==========================================
# 2. CỔNG GOOGLE CHAT BOT (CHUẨN HÓA 100%)
# ==========================================
def format_text_for_google_chat(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    return text.replace(r'\rightarrow', '➔').replace(r'$\rightarrow$', '➔').replace('$', '').strip()

async def call_fast_ai_google_chat(user_query: str) -> str:
    knowledge_context = get_smart_focused_knowledge(user_query, role="Sale")
    system_instruction = f"""
    Bạn là Trợ Lý KHO Sapo trên Google Chat.
    Nhiệm vụ: Trả lời thông minh, đi thẳng vào giải pháp, trình bày ngắn gọn kèm link Driver/Video bằng Markdown.
    
    KHO TRI THỨC KỸ THUẬT:
    {knowledge_context}
    """

    if GROQ_API_KEY and ACTIVE_GROQ_MODEL:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": ACTIVE_GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_query}
            ],
            "temperature": 0.1,
            "max_tokens": 800
        }
        try:
            res = await HTTP_CLIENT.post(url, headers=headers, json=payload, timeout=2.8)
            if res.status_code == 200:
                data = res.json()
                raw_reply = data["choices"][0]["message"]["content"]
                return format_text_for_google_chat(raw_reply)
        except Exception: pass

    # Dự phòng Gemini
    url_gemini = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={GEMINI_API_KEY}"
    payload_gemini = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": user_query}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 600}
    }
    try:
        res = await HTTP_CLIENT.post(url_gemini, json=payload_gemini, timeout=2.8)
        if res.status_code == 200:
            data = res.json()
            raw_reply = data["candidates"][0]["content"]["parts"][0]["text"]
            return format_text_for_google_chat(raw_reply)
    except Exception: pass
        
    return "❌ Hệ thống bận, anh/chị vui lòng gõ câu hỏi cụ thể hơn nhé."

@app.post("/google-chat")
async def google_chat_webhook(request: Request):
    try:
        event = await request.json()
        event_type = event.get("type")

        if event_type == "ADDED_TO_SPACE":
            return JSONResponse(content={"text": "👋 Xin chào! Tôi là *Trợ Lý KHO Sapo*. Hãy gõ câu hỏi kỹ thuật để tôi hỗ trợ ngay!"})

        if event_type == "MESSAGE":
            user_text = event.get("message", {}).get("text", "")
            # XÓA TOÀN BỘ THẺ THẦN THỦ MÃ HÓA CỦA GOOGLE CHAT
            cleaned_message = re.sub(r'<.*?>', '', user_text).replace("@Trợ Lý KHO Sapo", "").strip()

            quick_greetings = ["chào", "chào bạn", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em"]
            if not cleaned_message or cleaned_message.lower() in quick_greetings:
                return JSONResponse(content={
                    "text": "👋 Xin chào! Em là *Trợ Lý KHO Sapo*. Anh/chị cần hỗ trợ tra cứu thông số máy in, cài đặt driver hay khắc phục lỗi gì ạ?"
                })

            # KHÓA TIMEOUT CHỦ ĐỘNG 3.5 GIÂY ĐỂ GOOGLE CHAT KHÔNG BAO GIỜ BỊ LỖI
            try:
                ai_reply = await asyncio.wait_for(call_fast_ai_google_chat(cleaned_message), timeout=3.5)
                return JSONResponse(content={"text": ai_reply})
            except asyncio.TimeoutError:
                return JSONResponse(content={
                    "text": "⚡ *Trợ Lý KHO*: Vui lòng gõ câu hỏi ngắn gọn hơn (Ví dụ: `khổ tem SPL01` hoặc `cài máy in K200L`) để nhận đáp án ngay!"
                })

    except Exception as e:
        return JSONResponse(content={"text": f"❌ Lỗi hệ thống Bot: {str(e)}"})

    return JSONResponse(content={"text": "OK"})
