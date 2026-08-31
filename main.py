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

app = FastAPI(title="Trợ Lý KHO Perfect Engine", version="129.0")

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
        print("⚠️ Chưa có GROQ_API_KEY, sẽ dùng Gemini dự phòng.")
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
                "llama3-70b-8192"
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
    except Exception as e:
        print(f"⚠️ Cảnh báo nạp tab '{tab}': {e}")
    return tab, []

async def load_sheet_data_async():
    global RAM_CACHE
    tasks = [fetch_single_tab_raw(tab) for tab in ALL_TABS]
    results = await asyncio.gather(*tasks)
    RAM_CACHE = {tab: records for tab, records in results}
    print("✅ [CLOUD RUN] Đã nạp 100% dữ liệu RAM!")
    return {"status": "success"}

@app.get("/")
def health_check():
    return {
        "status": "healthy", 
        "active_groq_model": ACTIVE_GROQ_MODEL,
        "region": "asia-southeast1"
    }

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

def get_high_precision_knowledge(query: str, role: str) -> tuple[str, list]:
    accessible_tabs = ALL_TABS if role == "Sale" else TABS_PUBLIC
    query_lower = query.lower()
    
    stop_words = {"mình", "có", "bị", "được", "không", "cho", "với", "là", "và", "nhé", "ạ", "cần", "giúp", "tôi", "xin", "lỗi", "máy", "thế", "nào", "bao", "nhiêu", "thông", "số", "in", "qua", "đã", "ok"}
    words = [w for w in query_lower.split() if len(w) > 1 and w not in stop_words]
    if not words:
        words = [query_lower]

    scored_rows = []
    for tab in accessible_tabs:
        for row in RAM_CACHE.get(tab, []):
            row_text = " ".join(str(v).lower() for v in row.values())
            score = 0
            dev_name = str(row.get("Ten_Thiet_Bi", row.get("Loai_Thiet_Bi", ""))).lower()
            
            for w in words:
                if len(w) >= 3 and w in dev_name:
                    score += 50
                elif w in row_text:
                    score += 3
            
            if score > 0:
                scored_rows.append((score, tab, row))

    scored_rows.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored_rows[:2]

    if not top_matches:
        top_matches = [(1, "2_HUONG_DAN_CAI_DAT", r) for r in RAM_CACHE.get("2_HUONG_DAN_CAI_DAT", [])[:1]]

    knowledge_text = ""
    for score, tab, row in top_matches:
        knowledge_text += f"\n=== DỮ LIỆU CHUẨN TỪ TAB [{tab}] ===\n"
        for key, value in row.items():
            if value:
                knowledge_text += f"- {key}: {value}\n"
    return knowledge_text, [item[2] for item in top_matches]

def format_gchat_syntax(text: str) -> str:
    """ Chuyển đổi định dạng Markdown sang chuẩn Google Chat (<URL|Title> và *bold*) """
    # Chuyển **bold** thành *bold*
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    # Chuyển Markdown Link [Tên](URL) thành chuẩn Google Chat <URL|Tên>
    text = re.sub(r'\[(.*?)\]\((https?://.*?)\)', r'<\2|\1>', text)
    return text.replace(r'\rightarrow', '➔').replace(r'$\rightarrow$', '➔').replace('$', '').strip()

def format_direct_gchat_reply(matches: list) -> str:
    """ Trả về câu trả lời chuẩn Google Chat siêu tốc trong 0.001 giây """
    response = "🤖 *Trợ Lý KHO Sapo*:\n\n"
    for row in matches[:2]:
        dev_name = row.get("Ten_Thiet_Bi", row.get("Loai_Thiet_Bi", "Thiết bị Sapo"))
        guide = row.get("Noi_Dung_Huong_Dan", row.get("Cach_Khac_Phuc", row.get("Mo_Ta_Loi", row.get("Mo_Ta", ""))))
        driver = row.get("Link_Driver", row.get("Link_Video", ""))

        response += f"📌 *{dev_name}*\n"
        if guide:
            response += f"• *Hướng dẫn:* {guide}\n"
        if driver:
            response += f"• ➔ *Tài nguyên:* {format_gchat_syntax(driver)}\n"
        response += "\n"
    return response.strip()

def build_system_prompt(knowledge_context: str, role: str) -> str:
    return f"""
    Bạn là Trợ Lý KHO Sapo – Chuyên gia tư vấn kỹ thuật phần cứng Sapo.

    QUY TẮC PHẢN HỒI:
    1. KHÔNG TỰ VẼ BẢNG RÁC / BẢNG CHUẨN BỊ KHÔNG CÓ TRONG DỮ LIỆU.
    2. Liệt kê các phương pháp theo dạng gạch đầu dòng (`-` hoặc `•`) rõ ràng.
    3. Đính kèm ĐẦY ĐỦ link đúng cú pháp Markdown: `[Tên hiển thị](URL)`.
    4. Xưng "Trợ Lý KHO". Dùng `➔` chỉ hướng.

    PHÂN QUYỀN (ROLE: {role}):
    - 'Khach_Hang': Bảo hành Tab 4 bị khóa.
    - 'Sale': Mở khóa bảo hành nội bộ.

    DỮ LIỆU KỸ THUẬT:
    {knowledge_context}
    """

# ==========================================
# 1. CỔNG WEB VERCEL (/chat) - GROQ AI STREAMING
# ==========================================
@app.post("/chat")
async def chat_stream(req: ChatRequest):
    latest_msg = req.messages[-1]["text"] if req.messages else ""
    
    clean_q = re.sub(r'[^\w\s]', '', latest_msg.lower()).strip()
    quick_greetings = ["chào", "chào bạn", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em", "chào bạn nhé"]
    if clean_q in quick_greetings:
        async def greeting_gen():
            yield "Xin chào! Em là **Trợ Lý KHO Sapo**. Anh/chị cần hỗ trợ tra cứu thông số thiết bị hay cài đặt máy in nào ạ?"
        return StreamingResponse(greeting_gen(), media_type="text/plain")

    focused_knowledge, raw_matches = get_high_precision_knowledge(latest_msg, req.role)
    system_instruction = build_system_prompt(focused_knowledge, req.role)

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
            "temperature": 0.0,
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

            async for chunk in generate_gemini_stream(system_instruction, req.messages):
                yield chunk

        return StreamingResponse(generate_groq(), media_type="text/plain", headers={"Cache-Control": "no-cache"})

    return StreamingResponse(generate_gemini_stream(system_instruction, req.messages), media_type="text/plain", headers={"Cache-Control": "no-cache"})

async def generate_gemini_stream(system_instruction: str, messages: list):
    trimmed_messages = messages[-5:] if len(messages) > 5 else messages
    gemini_contents = []
    for m in trimmed_messages:
        role_type = "user" if m["role"] == "user" else "model"
        gemini_contents.append({"role": role_type, "parts": [{"text": m["text"]}]})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent?key={GEMINI_API_KEY}&alt=sse"
    headers = {"Content-Type": "application/json"}
    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": gemini_contents,
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1000}
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
# 2. CỔNG GOOGLE CHAT BOT (/google-chat) - RESPOND IN 0.001s
# ==========================================
@app.post("/google-chat")
async def google_chat_webhook(request: Request):
    try:
        event = await request.json()
        event_type = event.get("type")

        if event_type == "ADDED_TO_SPACE":
            return JSONResponse(content={"text": "👋 Xin chào! Tôi là *Trợ Lý KHO Sapo*. Hãy gõ câu hỏi kỹ thuật để tôi hỗ trợ ngay 24/7!"})

        if event_type == "MESSAGE":
            user_message = event.get("message", {}).get("text", "")
            cleaned_message = re.sub(r'<.*?>', '', user_message).replace("@Trợ Lý KHO Sapo", "").strip()

            quick_greetings = ["chào", "chào bạn", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em"]
            if not cleaned_message or cleaned_message.lower() in quick_greetings:
                return JSONResponse(content={
                    "text": "👋 Xin chào! Em là *Trợ Lý KHO Sapo*. Anh/chị cần hỗ trợ tra cứu thông số máy in, cài đặt driver hay khắc phục lỗi gì ạ?"
                })

            # Trả về câu trả lời định dạng chuẩn Google Chat ngay lập tức (0.001s)
            _, raw_matches = get_high_precision_knowledge(cleaned_message, role="Sale")
            reply_text = format_direct_gchat_reply(raw_matches)

            return JSONResponse(content={"text": reply_text})

    except Exception:
        return JSONResponse(content={"text": "👋 Trợ Lý KHO Sapo đã nhận thông tin. Bạn cần tra cứu thiết bị nào ạ?"})

    return JSONResponse(content={"text": "OK"})