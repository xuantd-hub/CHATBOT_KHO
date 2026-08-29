import os
import io
import json
import pandas as pd
import requests
import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Trợ Lý KHO Engine", version="12.0")

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

def load_sheet_data():
    global RAM_CACHE
    temp_cache = {}
    for tab in TABS:
        url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={tab}"
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200 and "text/csv" in res.headers.get("Content-Type", ""):
                df = pd.read_csv(io.BytesIO(res.content)).fillna("")
                records = df.to_dict(orient="records")
                cleaned_records = []
                for row in records:
                    cleaned_row = {k: str(v).strip() for k, v in row.items() if str(v).strip() != ""}
                    if cleaned_row:
                        cleaned_records.append(cleaned_row)
                temp_cache[tab] = cleaned_records
            else:
                temp_cache[tab] = []
        except Exception as e:
            print(f"⚠️ Cảnh báo tab '{tab}': {e}")
            temp_cache[tab] = []
            
    RAM_CACHE = temp_cache
    print("✅ Đã nạp thành công dữ liệu Google Sheet vào RAM!")
    return {"status": "success", "loaded_tabs": list(RAM_CACHE.keys())}

@app.on_event("startup")
def startup_event():
    load_sheet_data()

@app.get("/reload")
def reload_data():
    return load_sheet_data()

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
    Bạn là Trợ Lý KHO – Trợ lý tư vấn & chẩn đoán sự cố thiết bị phần cứng (máy in hóa đơn, máy quét mã vạch, máy POS, khay đựng tiền) chuyên nghiệp, cực kỳ thông minh và linh hoạt.

    QUY TẮC QUAN TRỌNG VỀ TÁC GIẢ & DANH TÍNH:
    1. Tên của bạn là "Trợ Lý KHO".
    2. TUYỆT ĐỐI KHÔNG tự ý chèn tên tác giả (Thái Đình Xuân / XuanTD) vào các câu chào hay hướng dẫn kỹ thuật.
    3. CHỈ KHI người dùng chủ động hỏi: "Ai tạo ra bạn?", "Tác giả là ai?", "Hệ thống này của ai?" thì bạn mới trả lời: "Hệ thống được phát triển bởi anh Thái Đình Xuân (XuanTD) - Leader Quản lý & Phát triển thiết bị."

    QUY TẮC PHẢN HỒI THÔNG MINH & TRÌNH BÀY ĐẸP MẮT:
    1. TUYỆT ĐỐI KHÔNG DÙNG LATEX: Không xuất các chuỗi `$\\rightarrow$`, `\\rightarrow`, `\\$`. Luôn dùng ký tự mũi tên Unicode chuẩn `➔` hoặc `->` khi hướng dẫn các bước click menu (Ví dụ: Control Panel ➔ Devices and Printers ➔ Printer Properties).
    2. TƯ DUY CHẨN ĐOÁN NỐI TIẾP:
       - Nếu người dùng hỏi mua/cài đặt ➔ Cung cấp các bước cài đặt + Link tải Driver / Video (nếu có).
       - Nếu người dùng báo lỗi tiếp theo (ví dụ: "vẫn báo đèn đỏ", "đã thử nhưng không in được") ➔ Chuyển ngay sang bước kiểm tra phần cứng (kẹt dao, hết giấy, đóng nắp máy) theo tư duy kỹ thuật chuyên sâu.
    3. TRÌNH BÀY RÕ RÀNG: Dùng các bước đánh số (Bước 1, Bước 2...) rõ ràng, in đậm các nút bấm quan trọng. Không ngắt câu giữa chừng.
    4. HÌNH ẢNH & LINK: Render chuẩn Markdown `[Tên hiển thị](URL)` cho link web/video, và `![Mô tả ảnh](URL)` cho ảnh minh họa.

    PHÂN QUYỀN VẬN HÀNH (ROLE: {req.role}):
    - 'Khach_Hang': Hướng dẫn kỹ thuật chuẩn xác, dễ làm theo. Nếu quá khả năng thì gợi ý liên hệ Tổng đài hỗ trợ kỹ thuật.
    - 'Sale': Cung cấp đầy đủ SĐT Kỹ thuật + Địa chỉ bảo hành kho HN và HCM (Tra trong Tab QUY_TRINH_LEO_THANG).

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

    # CỐ ĐỊNH CHUẨN MODEL GEMINI-3.6-FLASH DÀNH CHO PAID TIER
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:streamGenerateContent?key={GEMINI_API_KEY}&alt=sse"
    headers = {"Content-Type": "application/json"}
    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": gemini_contents,
        "generationConfig": {
            "temperature": 0.2,
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
