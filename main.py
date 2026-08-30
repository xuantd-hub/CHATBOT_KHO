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

app = FastAPI(title="Trợ Lý KHO Bulletproof Engine", version="114.0")

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

RAM_CACHE = {}

TABS_PUBLIC = [
    "1_THIET_BI_VA_LOI", 
    "2_HUONG_DAN_CAI_DAT", 
    "3_CHINH_SACH_SAPO", 
    "NHAN_DIEN_THIET_BI"
]
TAB_PRIVATE = "4_DU_LIEU_NOI_BO"
ALL_TABS = TABS_PUBLIC + [TAB_PRIVATE]

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
        timeout=httpx.Timeout(4.0, read=6.0),
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
        res = await HTTP_CLIENT.get(url, timeout=4.0)
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
    print("✅ [CLOUD RUN] Đã nạp 100% dữ liệu vào RAM!")
    return {"status": "success"}

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Trợ Lý KHO Bulletproof Engine"}

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

def get_smart_focused_knowledge(query: str, role: str) -> tuple[str, list]:
    stop_words = {"mình", "có", "bị", "được", "không", "cho", "với", "là", "và", "nhé", "ạ", "cần", "giúp", "tôi", "xin", "lỗi", "máy", "thế", "nào", "bao", "nhiêu", "thông", "số"}
    raw_words = [w.lower() for w in query.split() if len(w) > 1 and w.lower() not in stop_words]
    
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
    top_matches = scored_rows[:6]

    if not top_matches:
        top_matches = [(1, "2_HUONG_DAN_CAI_DAT", r) for r in RAM_CACHE.get("2_HUONG_DAN_CAI_DAT", [])[:3]]

    knowledge_text = ""
    for score, tab, row in top_matches:
        knowledge_text += f"\n--- [{tab}] ---\n"
        for key, value in row.items():
            knowledge_text += f"{key}: {value}\n"
    return knowledge_text, top_matches

# ==========================================
# 1. CỔNG WEB VERCEL (/chat) - KHÔNG BAO GIỜ TREO
# ==========================================
@app.post("/chat")
async def chat_stream(req: ChatRequest):
    latest_msg = req.messages[-1]["text"] if req.messages else ""
    focused_knowledge, _ = get_smart_focused_knowledge(latest_msg, req.role)

    system_instruction = f"""
    Bạn là Trợ Lý KHO Sapo – Chuyên gia tư vấn & kỹ thuật phần cứng Sapo.
    Nhiệm vụ: Trả lời ngắn gọn, rành mạch, đi thẳng vào đáp án dựa trên dữ liệu dưới đây.
    Đính kèm link Driver/Video bằng Markdown `[Tên](URL)`. Xưng danh "Trợ Lý KHO".

    KHO TRI THỨC KỸ THUẬT:
    {focused_knowledge}
    """

    async def generate_fast():
        # Gửi ký tự rỗng ngay lập tức để ngắt trạng thái chờ của Web
        yield ""
        
        # Thử Gemini 3.6 Flash trực tiếp
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:streamGenerateContent?key={GEMINI_API_KEY}&alt=sse"
        headers = {"Content-Type": "application/json"}
        payload = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": [{"text": latest_msg}]}],
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
                    return
        except Exception: pass

        # Nếu AI trục trặc, trả về tóm tắt trực tiếp từ Sheet trong 0.01s
        yield f"\n\n**Trợ Lý KHO phản hồi:**\n{focused_knowledge[:500]}"

    return StreamingResponse(generate_fast(), media_type="text/plain", headers={"Cache-Control": "no-cache"})

# ==========================================
# 2. CỔNG GOOGLE CHAT BOT (/google-chat) - PHẢN HỒI 100%
# ==========================================
def format_text_for_google_chat(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    return text.replace(r'\rightarrow', '➔').replace(r'$\rightarrow$', '➔').replace('$', '').strip()

async def get_ai_reply_quick(user_query: str, knowledge_context: str) -> str:
    system_instruction = f"""
    Bạn là Trợ Lý KHO Sapo trên Google Chat.
    Nhiệm vụ: Trả lời ngắn gọn dưới 3 dòng, kèm link Driver/Video bằng Markdown.
    DỮ LIỆU:
    {knowledge_context}
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": user_query}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 500}
    }
    res = await HTTP_CLIENT.post(url, json=payload, timeout=2.2)
    if res.status_code == 200:
        data = res.json()
        raw_reply = data["candidates"][0]["content"]["parts"][0]["text"]
        return format_text_for_google_chat(raw_reply)
    return ""

@app.post("/google-chat")
async def google_chat_webhook(request: Request):
    try:
        event = await request.json()
        event_type = event.get("type")

        if event_type == "ADDED_TO_SPACE":
            return JSONResponse(content={"text": "👋 Xin chào! Tôi là *Trợ Lý KHO Sapo*. Hãy gõ câu hỏi để tôi hỗ trợ ngay!"})

        if event_type == "MESSAGE":
            user_text = event.get("message", {}).get("text", "")
            cleaned_message = re.sub(r'<.*?>', '', user_text).replace("@Trợ Lý KHO Sapo", "").strip()

            quick_greetings = ["chào", "chào bạn", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em"]
            if not cleaned_message or cleaned_message.lower() in quick_greetings:
                return JSONResponse(content={
                    "text": "👋 Xin chào! Em là *Trợ Lý KHO Sapo*. Anh/chị cần hỗ trợ tra cứu thông số thiết bị hay cài đặt máy in gì ạ?"
                })

            knowledge_context, raw_matches = get_smart_focused_knowledge(cleaned_message, role="Sale")

            # Gọi AI với giới hạn thời gian 2.3 giây
            try:
                ai_reply = await asyncio.wait_for(get_ai_reply_quick(cleaned_message, knowledge_context), timeout=2.3)
                if ai_reply:
                    return JSONResponse(content={"text": ai_reply})
            except Exception: pass

            # DỰ PHÒNG SIÊU TỐC: Nếu AI quá 2.3s, tự bốc thông tin từ Sheet trả về lập tức (0.01s)
            fallback_text = "📌 *Thông tin tìm thấy trong Kho Tri Thức Sapo:*\n"
            for score, tab, row in raw_matches[:2]:
                fallback_text += f"\n• *{row.get('Ten_Thiet_Bi', 'Thiết bị')}*: {row.get('Mo_Ta_Loi', row.get('Mo_Ta', ''))}\n"
                if 'Link_Driver' in row and row['Link_Driver']:
                    fallback_text += f"  ➔ Driver: {row['Link_Driver']}\n"
            
            return JSONResponse(content={"text": format_text_for_google_chat(fallback_text)})

    except Exception as e:
        return JSONResponse(content={"text": f"❌ Lỗi hệ thống Bot: {str(e)}"})

    return JSONResponse(content={"text": "OK"})
