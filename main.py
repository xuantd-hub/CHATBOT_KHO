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

app = FastAPI(title="Trợ Lý KHO Master Cloud Engine", version="106.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SHEET_ID = os.getenv("SHEET_ID", "1ZMq0mTiQTDiP92UPaOIv39Q17WJXDiuvrcyYwfs7_Ag").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
SALE_SECRET_KEY = os.getenv("SALE_SECRET_KEY", "sapo2026").strip()

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
        timeout=httpx.Timeout(20.0, read=40.0),
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
    )
    await load_sheet_data_async()

@app.on_event("shutdown")
async def shutdown_event():
    if HTTP_CLIENT:
        await HTTP_CLIENT.aclose()

async def fetch_single_tab_raw(tab: str):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={tab}"
    try:
        res = await HTTP_CLIENT.get(url, timeout=10.0)
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
    return {"status": "healthy", "service": "Trợ Lý KHO Engine Full", "region": "asia-southeast1"}

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

def get_full_accessible_knowledge(role: str) -> str:
    accessible_tabs = ALL_TABS if role == "Sale" else TABS_PUBLIC
    full_data = {}
    for tab in accessible_tabs:
        full_data[tab] = RAM_CACHE.get(tab, [])
    return json.dumps(full_data, ensure_ascii=False, separators=(',', ':'))

# ==========================================
# 1. CỔNG DÀNH CHO WEB VERCEL (/chat)
# ==========================================
@app.post("/chat")
async def chat_stream(req: ChatRequest):
    full_knowledge_context = get_full_accessible_knowledge(req.role)

    system_instruction = f"""
    Bạn là Trợ Lý KHO – Chuyên gia tư vấn & kỹ thuật phần cứng Sapo. Thông minh, tận tâm và chính xác.
    NHIỆM VỤ:
    1. Đọc kỹ Kho Tri Thức bên dưới để giải thích rành mạch từng bước cho người dùng.
    2. Trích xuất ĐẦY ĐỦ link Driver/Video bằng Markdown `[Tên](URL)`.
    3. ZERO HALLUCINATION: Chỉ dùng dữ liệu có sẵn. Dùng `➔` chỉ hướng. Xưng danh "Trợ Lý KHO".

    BẢO MẬT (ROLE: {req.role}):
    - 'Khach_Hang': Dữ liệu bảo hành Tab 4 đã bị cắt 100%.
    - 'Sale': Mở khóa đầy đủ thông tin bảo hành nội bộ.

    KHO TRI THỨC TOÀN DIỆN:
    {full_knowledge_context}
    """
    trimmed_messages = req.messages[-5:] if len(req.messages) > 5 else req.messages
    gemini_contents = []
    for m in trimmed_messages:
        role_type = "user" if m["role"] == "user" else "model"
        gemini_contents.append({"role": role_type, "parts": [{"text": m["text"]}]})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:streamGenerateContent?key={GEMINI_API_KEY}&alt=sse"
    headers = {"Content-Type": "application/json"}
    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": gemini_contents,
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 3000}
    }

    async def generate():
        try:
            async with HTTP_CLIENT.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    err_body = await response.aread()
                    yield f"❌ Lỗi API Google ({response.status_code}): {err_body.decode('utf-8')}"
                    return
                async for line in response.aiter_lines():
                    if line and line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            data_json = json.loads(data_str)
                            if "candidates" in data_json and len(data_json["candidates"]) > 0:
                                chunk = data_json["candidates"][0]["content"]["parts"][0].get("text", "")
                                if chunk: yield chunk
                        except Exception:
                            pass
        except Exception as err:
            yield f"❌ Lỗi kết nối Google AI: {str(err)}"

    return StreamingResponse(generate(), media_type="text/plain", headers={"Cache-Control": "no-cache"})

# ==========================================
# 2. CỔNG DÀNH CHO GOOGLE CHAT BOT (/google-chat)
# ==========================================
def format_text_for_google_chat(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    return text.replace(r'\rightarrow', '➔').replace(r'$\rightarrow$', '➔').replace('$', '').strip()

async def ask_gemini_fast_google_chat(user_query: str) -> str:
    knowledge_context = get_full_accessible_knowledge(role="Sale")
    system_instruction = f"""
    Bạn là Trợ Lý KHO Sapo trên Google Chat nội bộ. Tận tâm, thông minh.
    Nhiệm vụ: Trả lời ngắn gọn, rành mạch dựa vào Kho Tri Thức.
    KHO TRI THỨC:
    {knowledge_context}
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": user_query}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1200}
    }
    try:
        res = await HTTP_CLIENT.post(url, json=payload, timeout=4.0)
        if res.status_code == 200:
            data = res.json()
            raw_reply = data["candidates"][0]["content"]["parts"][0]["text"]
            return format_text_for_google_chat(raw_reply)
    except Exception as e:
        return f"❌ Hệ thống bận: {str(e)}"
    return "❌ Dữ liệu không phản hồi."

@app.post("/google-chat")
async def google_chat_webhook(request: Request):
    try:
        event = await request.json()
        event_type = event.get("type")

        if event_type == "ADDED_TO_SPACE":
            return JSONResponse(content={"text": "👋 Xin chào! Tôi là *Trợ Lý KHO Sapo*. Hãy gõ câu hỏi để tôi hỗ trợ ngay!"})

        if event_type == "MESSAGE":
            user_message = event.get("message", {}).get("text", "")
            cleaned_message = user_message.replace("@Trợ Lý KHO Sapo", "").strip()

            if not cleaned_message or cleaned_message.lower() in ["chào", "chào bạn", "hi", "hello"]:
                return JSONResponse(content={"text": "👋 Xin chào! Em là *Trợ Lý KHO Sapo*. Anh/chị cần hỗ trợ gì ạ?"})

            ai_reply = await ask_gemini_fast_google_chat(cleaned_message)
            return JSONResponse(content={"text": ai_reply})

    except Exception as e:
        return JSONResponse(content={"text": f"❌ Lỗi hệ thống Bot: {str(e)}"})

    return JSONResponse(content={"text": "OK"})
