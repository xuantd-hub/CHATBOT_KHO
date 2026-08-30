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

app = FastAPI(title="Trợ Lý KHO Sapo Master Pro", version="122.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SHEET_ID = os.getenv("SHEET_ID", "1ZMq0mTiQTDiP92UPaOIv39Q17WJXDiuvrcyYwfs7_Ag").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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").strip()
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
        timeout=httpx.Timeout(5.0, read=8.0),
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
    total_items = sum(len(v) for v in RAM_CACHE.values())
    print(f"✅ [RAM CACHE] Đã nạp thành công {total_items} dòng dữ liệu từ Google Sheet!")
    return {"status": "success"}

@app.get("/")
def health_check():
    total_items = sum(len(v) for v in RAM_CACHE.values())
    return {
        "status": "healthy",
        "service": "Trợ Lý KHO Sapo Master Pro",
        "total_records": total_items
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

def get_smart_knowledge_context(query: str, role: str) -> str:
    """ Tìm kiếm chính xác, ưu tiên các dòng chứa nhiều từ khóa khớp nhất """
    accessible_tabs = ALL_TABS if role == "Sale" else TABS_PUBLIC
    words = [w.lower() for w in query.split() if len(w) > 1]
    
    scored_rows = []
    for tab in accessible_tabs:
        for row in RAM_CACHE.get(tab, []):
            row_text = " ".join(str(v).lower() for v in row.values())
            score = sum(2 if w in row_text else 0 for w in words)
            # Thưởng điểm nếu tên thiết bị khớp trực tiếp
            dev_name = str(row.get("Ten_Thiet_Bi", row.get("Loai_Thiet_Bi", ""))).lower()
            if any(w in dev_name for w in words):
                score += 5
            
            if score > 0:
                scored_rows.append((score, tab, row))

    scored_rows.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored_rows[:4]  # Lấy top 4 dòng chuẩn xác nhất để AI đọc kỹ

    if not top_matches:
        top_matches = [(1, "2_HUONG_DAN_CAI_DAT", r) for r in RAM_CACHE.get("2_HUONG_DAN_CAI_DAT", [])[:2]]

    knowledge_text = ""
    for score, tab, row in top_matches:
        knowledge_text += f"\n--- NGUỒN DỮ LIỆU TỪ TAB [{tab}] ---\n"
        for key, value in row.items():
            knowledge_text += f"{key}: {value}\n"
    return knowledge_text

def build_system_prompt(knowledge_context: str, role: str) -> str:
    return f"""
    Bạn là Trợ Lý KHO Sapo – Chuyên gia kỹ thuật phần cứng chuyên sâu, nhạy bén và cực kỳ chính xác của công ty Sapo.

    QUY TẮC BẮT BUỘC KHI TRẢ LỜI:
    1. **Bám sát tuyệt đối dữ liệu được cung cấp:** Không được bịa đặt thông tin ngoài dữ liệu. Nếu dữ liệu có hướng dẫn cụ thể, hãy bóc tách và trình bày lại thật rành mạch, chi tiết từng bước.
    2. **Trả lời thẳng vào trọng tâm:** Không nói lan man, không chào hỏi dài dòng sướt mướt. Đi thẳng vào giải pháp, thông số hoặc hướng dẫn xử lý lỗi mà khách hàng đang hỏi.
    3. **Định dạng Markdown chuyên nghiệp:** Dùng tiêu đề phụ (`###`), gạch đầu dòng (`-`), in đậm các thông số quan trọng (tên thiết bị, mã lỗi, phím bấm).
    4. **Trích xuất đủ Link:** Nếu trong dữ liệu có chứa link (Drive, Video, Hướng dẫn), bắt buộc phải gắn link dưới dạng Markdown chuẩn: `[Tên hiển thị](URL)`.
    5. Xưng "Trợ Lý KHO". Dùng mũi tên `➔` để chỉ hướng thao tác.

    PHÂN QUYỀN BẢO MẬT (ROLE: {role}):
    - 'Khach_Hang': Tuyệt đối KHÔNG tiết lộ thông tin bảo hành nội bộ hoặc quy trình đổi trả nội bộ từ Tab 4.
    - 'Sale': Được phép giải đáp đầy đủ thông tin bảo hành, xử lý nội bộ từ Tab 4.

    KHO TRI THỨC KĨ THUẬT CHUẨN XÁC:
    {knowledge_context}
    """

# ==========================================
# 1. CỔNG WEB VERCEL (/chat) - GROQ STREAM 0.3S
# ==========================================
@app.post("/chat")
async def chat_stream(req: ChatRequest):
    latest_msg = req.messages[-1]["text"] if req.messages else ""
    
    # Xử lý nhanh câu chào
    clean_q = re.sub(r'[^\w\s]', '', latest_msg.lower()).strip()
    quick_greetings = ["chào", "chào bạn", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em", "chào bạn nhé"]
    if clean_q in quick_greetings:
        async def greeting_gen():
            yield "Xin chào! Em là **Trợ Lý KHO Sapo**. Anh/chị cần hỗ trợ tra cứu thông số thiết bị, cài đặt máy in hay xử lý lỗi cụ thể nào ạ?"
        return StreamingResponse(greeting_gen(), media_type="text/plain")

    focused_knowledge = get_smart_knowledge_context(latest_msg, req.role)
    system_instruction = build_system_prompt(focused_knowledge, req.role)

    async def generate_groq():
        yield ""  # Mở luồng ngay lập tức
        
        groq_models = ["llama-3.1-8b-instant", "llama3-8b-8192", "llama-3.3-70b-versatile"]
        for g_model in groq_models:
            try:
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
                payload = {
                    "model": g_model,
                    "messages": [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": latest_msg}
                    ],
                    "temperature": 0.1,  # Đặt nhiệt độ cực thấp để AI tập trung tối đa vào độ chính xác, không sáng tác bừa
                    "max_tokens": 1000,
                    "stream": True
                }
                async with HTTP_CLIENT.stream("POST", url, headers=headers, json=payload) as res:
                    if res.status_code == 200:
                        async for line in res.aiter_lines():
                            if line and line.startswith("data: "):
                                d_str = line[6:].strip()
                                if d_str == "[DONE]": break
                                try:
                                    d_json = json.loads(d_str)
                                    choices = d_json.get("choices", [])
                                    if choices:
                                        chunk = choices[0].get("delta", {}).get("content", "")
                                        if chunk: yield chunk
                                except Exception: pass
                        return
            except Exception: continue

        yield "❌ Trợ Lý KHO đang bận kết nối, anh/chị vui lòng gửi lại câu hỏi ngắn gọn hơn nhé."

    return StreamingResponse(generate_groq(), media_type="text/plain", headers={"Cache-Control": "no-cache"})

# ==========================================
# 2. CỔNG GOOGLE CHAT BOT (/google-chat)
# ==========================================
def format_for_gchat(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    return text.replace(r'\rightarrow', '➔').replace(r'$\rightarrow$', '➔').replace('$', '').strip()

async def call_groq_fast_gchat(user_query: str, role: str = "Sale") -> str:
    focused_knowledge = get_smart_knowledge_context(user_query, role)
    system_instruction = build_system_prompt(focused_knowledge, role)

    groq_models = ["llama-3.1-8b-instant", "llama3-8b-8192", "llama-3.3-70b-versatile"]
    for g_model in groq_models:
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": g_model,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_query}
                ],
                "temperature": 0.1,
                "max_tokens": 800
            }
            res = await HTTP_CLIENT.post(url, headers=headers, json=payload, timeout=3.2)
            if res.status_code == 200:
                data = res.json()
                raw_text = data["choices"][0]["message"]["content"]
                return format_for_gchat(raw_text)
        except Exception: continue

    return "❌ Hệ thống đang bận, anh/chị thử lại câu hỏi nhé."

@app.post("/google-chat")
async def google_chat_webhook(request: Request):
    try:
        event = await request.json()
        event_type = event.get("type")

        if event_type == "ADDED_TO_SPACE":
            return JSONResponse(content={"text": "👋 Xin chào! Tôi là *Trợ Lý KHO Sapo*. Hãy gõ câu hỏi kỹ thuật để tôi hỗ trợ ngay!"})

        if event_type == "MESSAGE":
            user_text = event.get("message", {}).get("text", "")
            cleaned_message = re.sub(r'<.*?>', '', user_text).replace("@Trợ Lý KHO Sapo", "").strip()
            clean_q = re.sub(r'[^\w\s]', '', cleaned_message.lower()).strip()

            quick_greetings = ["chào", "chào bạn", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em", "chào bạn nhé"]
            if not clean_q or clean_q in quick_greetings:
                return JSONResponse(content={
                    "text": "👋 Xin chào! Em là *Trợ Lý KHO Sapo*. Anh/chị cần tra cứu thông tin thiết bị hay hướng dẫn cài đặt nào ạ?"
                })

            ai_reply = await call_groq_fast_gchat(cleaned_message, role="Sale")
            return JSONResponse(content={"text": ai_reply})

    except Exception as e:
        return JSONResponse(content={"text": f"❌ Lỗi xử lý Bot: {str(e)}"})

    return JSONResponse(content={"text": "OK"})
