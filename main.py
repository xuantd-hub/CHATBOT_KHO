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

app = FastAPI(title="Trợ Lý KHO Engine", version="15.0")

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
        res = await client.get(url, timeout=8.0)
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
        
    temp_cache = {tab: records for tab, records in results}
    RAM_CACHE = temp_cache
    print("✅ Đã nạp song song siêu tốc dữ liệu Google Sheet vào RAM!")
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
            filtered_data[tab_name] = matched_rows[:5]

    if not filtered_data:
        filtered_data = {
            "LUONG_CHAN_DOAN": RAM_CACHE.get("LUONG_CHAN_DOAN", [])[:3],
            "QUY_TRINH_CHUNG": RAM_CACHE.get("QUY_TRINH_CHUNG", [])[:2],
            "QUY_TRINH_LEO_THANG": RAM_CACHE.get("QUY_TRINH_LEO_THANG", [])[:2]
        }

    return json.dumps(filtered_data, ensure_ascii=False, separators=(',', ':'))

@app.post("/chat")
async def chat_stream(req: ChatRequest):
    latest_msg = req.messages[-1]["text"] if req.messages else ""
    compact_knowledge = filter_relevant_knowledge(latest_msg)

    system_instruction = f"""
    Bạn là Trợ Lý KHO – Trợ lý tư vấn & chẩn đoán sự cố thiết bị phần cứng chuyên nghiệp của Sapo.

    WEBSITE THAM CHIẾU CHÍNH THỨC:
    - Trang chủ chính: https://sapo.vn
    - Trang thiết bị phần cứng: https://shop.sapo.vn

    QUY TẮC PHẢN HỒI NỘI DUNG (BẮT BUỘC CHÍNH XÁC & ĐẦY ĐỦ 100%):
    1. TRÍCH XUẤT ĐẦY ĐỦ CHI TIẾT TỪNG BƯỚC:
       - Khi người dùng hỏi hướng dẫn cài đặt hay chẩn đoán lỗi (ví dụ: K200L, SPR02...): BẮT BUỘC phải trích xuất và trình bày TOÀN BỘ chi tiết có trong 'Nội dung hướng dẫn' hoặc 'Thao tác thực hiện' của Kho Tri Thức.
       - Mô tả chi tiết từng thao tác: nhấn nút nào, giữ bao nhiêu giây, bật/tắt công tắc, chọn tab/mục nào trong Windows/Mac, chọn cổng kết nối (USB/LAN), chọn khổ giấy (80mm/XP-80) và các lưu ý kỹ thuật.
       - TUYỆT ĐỐI KHÔNG tự ý tóm tắt ngắn gọn.
    2. CHỐNG BỊA THÔNG TIN BẢO HÀNH / ĐỊA CHỈ:
       - CHỈ CUNG CẤP địa chỉ bảo hành, SĐT khi thông tin đó CÓ TRONG KHO TRI THỨC (Google Sheet).
       - Nếu thông tin chưa có trong Sheet, báo rõ chưa cập nhật và hướng dẫn truy cập https://shop.sapo.vn hoặc liên hệ Tổng đài Sapo.
    3. QUY TẮC TRÌNH BÀY & TÁC GIẢ:
       - TUYỆT ĐỐI KHÔNG DÙNG LATEX (`$\\rightarrow$`, `\\rightarrow`, `\\$`). Dùng ký tự Unicode `➔` hoặc `->` khi hướng dẫn bấm menu.
       - Tên của bạn là "Trợ Lý KHO". Không tự chèn tên tác giả vào câu chào.
       - Chỉ khi người dùng hỏi "Ai tạo ra bạn?", "Tác giả là ai?" thì mới trả lời: "Hệ thống được phát triển bởi anh Thái Đình Xuân (XuanTD) - Nhân viên Quản lý & Phát triển thiết bị."

    PHÂN QUYỀN VẬN HÀNH (ROLE: {req.role}):
    - 'Khach_Hang': Hướng dẫn kỹ thuật chuẩn xác, cực kỳ chi tiết, từng bước dễ thao tác.
    - 'Sale': Cung cấp thông tin quy trình bảo hành theo dữ liệu sẵn có trong Sheet.

    KHO TRI THỨC TRA CỨU:
    {compact_knowledge}
    """

    gemini_contents = []
    for m in req.messages:
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
            "maxOutputTokens": 2048
        }
    }

    async def generate():
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=60.0)) as client:
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
