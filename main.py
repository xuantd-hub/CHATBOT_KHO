import os
import io
import json
import asyncio
import pandas as pd
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Trợ Lý KHO Enterprise Cloud & Google Chat", version="104.0")

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

def get_focused_knowledge(query: str, role: str = "Sale") -> str:
    """ Trích xuất đúng các khối liên quan để đảm bảo phản hồi dưới 1.5s """
    stop_words = {"mình", "có", "bị", "được", "không", "cho", "với", "là", "và", "nhé", "ạ", "cần", "giúp", "tôi", "xin", "lỗi", "máy", "thế", "nào", "bao", "nhiêu"}
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
    top_matches = scored_rows[:6]

    if not top_matches:
        # Nếu không tìm thấy từ khóa khớp, nạp tab Hướng dẫn cài đặt
        top_matches = [(1, "2_HUONG_DAN_CAI_DAT", r) for r in RAM_CACHE.get("2_HUONG_DAN_CAI_DAT", [])[:5]]

    knowledge_text = ""
    for score, tab, row in top_matches:
        knowledge_text += f"\n--- [Nguồn: {tab}] ---\n"
        for key, value in row.items():
            knowledge_text += f"{key}: {value}\n"
    return knowledge_text

async def ask_gemini_fast(user_query: str) -> str:
    """Xử lý đồng bộ siêu tốc dưới 2 giây cho Google Chat"""
    knowledge_context = get_focused_knowledge(user_query, role="Sale")
    system_instruction = f"""
    Bạn là Trợ Lý KHO Sapo trên Google Chat nội bộ. Tận tâm, thông minh và chính xác.
    Nhiệm vụ: Trả lời ngắn gọn, rành mạch từng bước dựa vào Kho Tri Thức được trích xuất dưới đây. Cung cấp đầy đủ link Driver/Video bằng Markdown.
    
    KHO TRI THỨC TRÍCH XUẤT:
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
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"Lỗi Gemini: {e}")
        return "❌ Hệ thống phản hồi chậm, vui lòng thử lại câu hỏi cụ thể hơn."
    return "❌ Không thể lấy dữ liệu từ AI."

# ==========================================
# ENDPOINT DÀNH RIÊNG CHO GOOGLE CHAT BOT
# ==========================================
@app.post("/google-chat")
async def google_chat_webhook(request: Request):
    try:
        event = await request.json()
        event_type = event.get("type")

        if event_type == "ADDED_TO_SPACE":
            return JSONResponse({
                "text": "👋 Xin chào! Tôi là **Trợ Lý KHO Sapo**. Hãy gõ mã thiết bị hoặc câu hỏi kỹ thuật để tôi hỗ trợ ngay 24/7!"
            })

        if event_type == "MESSAGE":
            user_message = event.get("message", {}).get("text", "")
            cleaned_message = user_message.replace("@Trợ Lý KHO Sapo", "").strip()
            
            if not cleaned_message or cleaned_message.lower() in ["chào", "chào bạn", "hi", "hello"]:
                return JSONResponse({
                    "text": "👋 Xin chào! Em là **Trợ Lý KHO Sapo**. Anh/chị cần hỗ trợ kiểm tra thiết bị, cài đặt máy in hay chẩn đoán lỗi gì ạ?"
                })

            # Gọi xử lý siêu tốc
            ai_reply = await ask_gemini_fast(cleaned_message)
            return JSONResponse({"text": ai_reply})

    except Exception as e:
        return JSONResponse({"text": f"❌ Lỗi xử lý Bot: {str(e)}"})

    return JSONResponse({"text": "OK"})
