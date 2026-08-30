import os
import io
import json
import asyncio
import pandas as pd
import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Trợ Lý KHO Engine", version="16.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SHEET_ID = "1ZMq0mTiQTDiP92UPaOIv39Q17WJXDiuvrcyYwfs7_Ag"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

RAM_CACHE = {}
TABS = [
    "DANH_MUC_LOI", "THAO_TAC_CAI_DAT", "LUONG_CHAN_DOAN", 
    "NHAN_DIEN_THIET_BI", "QUY_TRINH_CHUNG", "DANH_MUC_SAN_PHAM", 
    "QUY_TRINH_LEO_THANG", "NHAN_SU_THIET_BI"
]

async def fetch_single_tab(client: httpx.AsyncClient, tab: str):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={tab}"
    try:
        res = await client.get(url, timeout=6.0)
        if res.status_code == 200 and "text/csv" in res.headers.get("Content-Type", ""):
            df = pd.read_csv(io.BytesIO(res.content)).fillna("")
            records = df.to_dict(orient="records")
            cleaned_records = []
            for row in records:
                cleaned_row = {k: str(v).strip() for k, v in row.items() if str(v).strip() != ""}
                if cleaned_row:
                    cleaned_records.append(cleaned_row)
            return tab, cleaned_records
    except Exception as e:
        print(f"⚠️ Cảnh báo tab '{tab}': {e}")
    return tab, []

async def load_sheet_data_async():
    global RAM_CACHE
    async with httpx.AsyncClient() as client:
        tasks = [fetch_single_tab(client, tab) for tab in TABS]
        results = await asyncio.gather(*tasks)
        
    RAM_CACHE = {tab: records for tab, records in results}
    print("✅ Đã nạp dữ liệu Google Sheet vào RAM!")
    return {"status": "success", "loaded_tabs": list(RAM_CACHE.keys())}

@app.on_event("startup")
async def startup_event():
    await load_sheet_data_async()

@app.get("/reload")
async def reload_data():
    return await load_sheet_data_async()

class ChatRequest(BaseModel):
    messages: list
    role: str = "Khach_Hang"

def filter_relevant_knowledge(latest_user_msg: str) -> str:
    query_words = [w.lower() for w in latest_user_msg.split() if len(w) > 1]
    filtered_data = {}

    for tab_name, rows in RAM_CACHE.items():
        matched_rows = []
        for row in rows:
            row_text = " ".join(row.values()).lower()
            if any(word in row_text for word in query_words):
                matched_rows.append(row)
        if matched_rows:
            filtered_data[tab_name] = matched_rows[:3]

    if not filtered_data:
        filtered_data = {
            "LUONG_CHAN_DOAN": RAM_CACHE.get("LUONG_CHAN_DOAN", [])[:2],
            "QUY_TRINH_CHUNG": RAM_CACHE.get("QUY_TRINH_CHUNG", [])[:2]
        }

    return json.dumps(filtered_data, ensure_ascii=False, separators=(',', ':'))

@app.post("/chat")
async def chat_stream(req: ChatRequest):
    latest_msg = req.messages[-1]["text"] if req.messages else ""
    compact_knowledge = filter_relevant_knowledge(latest_msg)

    system_instruction = f"""
    Bạn là Trợ Lý KHO – Trợ lý tư vấn & chẩn đoán sự cố thiết bị phần cứng chuyên nghiệp của Sapo.

    WEBSITE CHÍNH THỨC: https://sapo.vn | THIẾT BỊ: https://shop.sapo.vn

    QUY TẮC PHẢN HỒI NỘI DUNG (CHÍNH XÁC & HOÀN CHỈNH 100%):
    1. HOÀN THÀNH ĐẦY ĐỦ CÂU VÀ LINK MARKDOWN:
       - Khi xuất link hoặc video, BẮT BUỘC phải viết trọn vẹn cú pháp Markdown dạng `[Tên hiển thị](URL)`. TUYỆT ĐỐI KHÔNG ngắt câu hay bỏ dở link giữa chừng.
       - Trích xuất ĐẦY ĐỦ toàn bộ nội dung hướng dẫn thao tác kỹ thuật từ Kho Tri Thức (nhấn nút nào, giữ bao nhiêu giây, cổng kết nối, khổ giấy).
    2. NGUYÊN TẮC CHỐNG BỊA ĐỊA CHỈ:
       - CHỈ CUNG CẤP địa chỉ bảo hành/SĐT khi thông tin đó CÓ TRONG KHO TRI THỨC.
       - Nếu thông tin chưa có, báo rõ chưa cập nhật và hướng dẫn truy cập https://shop.sapo.vn hoặc liên hệ Tổng đài Sapo.
    3. ĐỊNH DẠNG TRÌNH BÀY:
       - TUYỆT ĐỐI KHÔNG DÙNG LATEX (`$\\rightarrow$`, `\\rightarrow`, `\\$`). Dùng ký tự Unicode `➔` hoặc `->` khi hướng dẫn bấm menu.
       - Tên của bạn là "Trợ Lý KHO". Chỉ khi người dùng hỏi "Ai tạo ra bạn?" mới nêu tên tác giả Thái Đình Xuân (XuanTD).

    PHÂN QUYỀN: 'Khach_Hang' (Hướng dẫn kỹ thuật chi tiết), 'Sale' (Thông tin quy trình bảo hành).

    KHO TRI THỨC TRA CỨU:
    {compact_knowledge}
    """

    # Chỉ giữ 6 tin nhắn gần nhất để tối ưu tốc độ xử lý
    trimmed_messages = req.messages[-6:] if len(req.messages) > 6 else req.messages
    gemini_contents = []
    for m in trimmed_messages:
        role_type = "user" if m["role"] == "user" else "model"
        gemini_contents.append({
            "role": role_type,
            "parts": [{"text": m["text"]}]
        })

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:streamGenerateContent?key={GEMINI_API_KEY}&alt=sse"
    headers = {"Content-Type": "application/json"}
    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": gemini_contents,
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 2500
        }
    }

    async def generate():
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, read=40.0)) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    err_body = await response.aread()
                    yield f"❌ Lỗi {response.status_code}: {err_body.decode('utf-8')}"
                    return

                async for line in response.aiter_lines():
                    if line and line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            data_json = json.loads(data_str)
                            chunk = data_json['candidates'][0]['content']['parts'][0]['text']
                            yield chunk
                        except Exception:
                            pass

    return StreamingResponse(
        generate(), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )
