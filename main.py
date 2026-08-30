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

app = FastAPI(title="Trợ Lý KHO Groq UltraFast Engine", version="111.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SHEET_ID = os.getenv("SHEET_ID", "1ZMq0mTiQTDiP92UPaOIv39Q17WJXDiuvrcyYwfs7_Ag").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
SALE_SECRET_KEY = os.getenv("SALE_SECRET_KEY", "sapo2026").strip()

# Mô hình Llama 3.3 70B Versatile siêu tốc trên chip Groq LPU
MODEL_NAME = "llama-3.3-70b-versatile"

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
        timeout=httpx.Timeout(8.0, read=10.0),
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
        res = await HTTP_CLIENT.get(url, timeout=6.0)
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
    print("✅ [GROQ ENGINE] Đã nạp 100% dữ liệu Google Sheet vào RAM!")
    return {"status": "success"}

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Trợ Lý KHO Groq Engine", "region": "asia-southeast1"}

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

def get_focused_knowledge(query: str, role: str) -> str:
    stop_words = {"mình", "có", "bị", "được", "không", "cho", "với", "là", "và", "nhé", "ạ", "cần", "giúp", "tôi", "xin", "lỗi", "máy", "thế", "nào", "bao", "nhiêu", "thông", "số"}
    words = [w.lower() for w in query.split() if len(w) > 1 and w.lower() not in stop_words]
    if not words:
        words = [query.lower()]

    accessible_tabs = ALL_TABS if role == "Sale" else TABS_PUBLIC
    scored_rows = []

    for tab in accessible_tabs:
        for row in RAM_CACHE.get(tab, []):
            row_text = " ".join(str(v).lower() for v in row.values())
            score = sum(1 for w in words if w in row_text)
            model_name = str(row.get("Ten_Thiet_Bi", "")).lower()
            if any(w in model_name for w in words):
                score += 10
            if score > 0:
                scored_rows.append((score, tab, row))

    scored_rows.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored_rows[:5]

    if not top_matches:
        top_matches = [(1, "2_HUONG_DAN_CAI_DAT", r) for r in RAM_CACHE.get("2_HUONG_DAN_CAI_DAT", [])[:3]]

    knowledge_text = ""
    for score, tab, row in top_matches:
        knowledge_text += f"\n--- [{tab}] ---\n"
        for key, value in row.items():
            knowledge_text += f"{key}: {value}\n"
    return knowledge_text

# ==========================================
# 1. CỔNG WEB VERCEL (/chat) - GROQ STREAM (0.2S)
# ==========================================
@app.post("/chat")
async def chat_stream(req: ChatRequest):
    latest_msg = req.messages[-1]["text"] if req.messages else ""
    focused_knowledge = get_focused_knowledge(latest_msg, req.role)

    system_instruction = f"""
    Bạn là Trợ Lý KHO Sapo – Chuyên gia kỹ thuật & thiết bị kho.
    NHIỆM VỤ:
    1. Trả lời rành mạch, chính xác dựa trên Kho Tri Thức dưới đây.
    2. Cung cấp ĐẦY ĐỦ link Driver/Video bằng Markdown `[Tên](URL)`.
    3. ZERO HALLUCINATION: Chỉ dùng thông tin trong dữ liệu. Dùng `➔` chỉ hướng. Xưng danh "Trợ Lý KHO".

    BẢO MẬT (ROLE: {req.role}):
    - 'Khach_Hang': Dữ liệu bảo hành Tab 4 đã bị khóa 100%.
    - 'Sale': Mở khóa thông tin bảo hành nội bộ.

    KHO TRI THỨC KỸ THUẬT:
    {focused_knowledge}
    """

    messages_payload = [{"role": "system", "content": system_instruction}]
    trimmed = req.messages[-5:] if len(req.messages) > 5 else req.messages
    for m in trimmed:
        role_type = "user" if m["role"] == "user" else "assistant"
        messages_payload.append({"role": role_type, "content": m["text"]})

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages_payload,
        "temperature": 0.1,
        "max_tokens": 1000,
        "stream": True
    }

    async def generate():
        try:
            async with HTTP_CLIENT.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    err_body = await response.aread()
                    yield f"❌ Lỗi Groq API ({response.status_code}): {err_body.decode('utf-8')}"
                    return
                async for line in response.aiter_lines():
                    if line and line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            choices = data_json.get("choices", [])
                            if choices:
                                chunk = choices[0].get("delta", {}).get("content", "")
                                if chunk:
                                    yield chunk
                        except Exception:
                            pass
        except Exception as err:
            yield f"❌ Lỗi kết nối Groq Engine: {str(err)}"

    return StreamingResponse(generate(), media_type="text/plain", headers={"Cache-Control": "no-cache"})

# ==========================================
# 2. CỔNG GOOGLE CHAT BOT (/google-chat) - GROQ FAST (0.4S)
# ==========================================
def format_text_for_google_chat(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    return text.replace(r'\rightarrow', '➔').replace(r'$\rightarrow$', '➔').replace('$', '').strip()

async def call_groq_fast_google_chat(user_query: str) -> str:
    knowledge_context = get_focused_knowledge(user_query, role="Sale")
    system_instruction = f"""
    Bạn là Trợ Lý KHO Sapo trên Google Chat nội bộ.
    Nhiệm vụ: Trả lời ngắn gọn, rành mạch, đi thẳng vào đáp án và đính kèm link Driver/Video bằng Markdown.
    
    KHO TRI THỨC KỸ THUẬT:
    {knowledge_context}
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_query}
        ],
        "temperature": 0.1,
        "max_tokens": 800,
        "stream": False
    }
    try:
        res = await HTTP_CLIENT.post(url, headers=headers, json=payload, timeout=3.5)
        if res.status_code == 200:
            data = res.json()
            raw_reply = data["choices"][0]["message"]["content"]
            return format_text_for_google_chat(raw_reply)
    except Exception as e:
        print(f"Lỗi Groq Fast: {e}")
        return "❌ Hệ thống đang bận, vui lòng thử lại với câu hỏi cụ thể hơn."
    return "❌ Dữ liệu không phản hồi từ Groq Engine."

@app.post("/google-chat")
async def google_chat_webhook(request: Request):
    try:
        event = await request.json()
        event_type = event.get("type")

        if event_type == "ADDED_TO_SPACE":
            return JSONResponse(content={"text": "👋 Xin chào! Tôi là *Trợ Lý KHO Sapo* (Groq UltraFast). Hãy gõ câu hỏi kỹ thuật để tôi hỗ trợ ngay 24/7!"})

        if event_type == "MESSAGE":
            user_message = event.get("message", {}).get("text", "")
            cleaned_message = user_message.replace("@Trợ Lý KHO Sapo", "").strip()

            quick_greetings = ["chào", "chào bạn", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em"]
            if not cleaned_message or cleaned_message.lower() in quick_greetings:
                return JSONResponse(content={
                    "text": "👋 Xin chào! Em là *Trợ Lý KHO Sapo*. Anh/chị cần hỗ trợ tra cứu thông số máy in, cài đặt driver hay khắc phục lỗi gì ạ?"
                })

            ai_reply = await call_groq_fast_google_chat(cleaned_message)
            return JSONResponse(content={"text": ai_reply})

    except Exception as e:
        return JSONResponse(content={"text": f"❌ Lỗi hệ thống Bot: {str(e)}"})

    return JSONResponse(content={"text": "OK"})
