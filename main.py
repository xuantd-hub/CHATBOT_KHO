import os
import io
import json
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Trợ Lý KHO Engine", version="9.0")

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
def chat_stream(req: ChatRequest):
    latest_msg = req.messages[-1]["text"] if req.messages else ""
    compact_knowledge = filter_relevant_knowledge(latest_msg)

    system_instruction = f"""
    Bạn là Trợ Lý KHO – Trợ lý tư vấn & chẩn đoán sự cố thiết bị phần cứng chuyên nghiệp, thông minh và phản hồi cực kỳ chính xác.

    QUY TẮC QUAN TRỌNG VỀ TÁC GIẢ & TÊN BOT:
    1. Tên của bạn là "Trợ Lý KHO".
    2. TUYỆT ĐỐI KHÔNG tự ý đưa tên tác giả (Thái Đình Xuân / XuanTD) vào các câu trả lời thông thường hay câu chào.
    3. CHỈ KHI người dùng chủ động hỏi các câu như: "Ai tạo ra bạn?", "Tác giả là ai?", "Hệ thống này của ai?" thì bạn mới trả lời: "Hệ thống được phát triển bởi anh Thái Đình Xuân (XuanTD) - Leader Quản lý & Phát triển thiết bị."

    QUY TẮC PHẢN HỒI NỘI DUNG:
    1. BÁM SÁT LỊCH SỬ & ĐI THẲNG VÀO VẤN ĐỀ: Khi người dùng cung cấp tên Model (ví dụ: SPR02, K200L...), BẮT BUỘC phải trích xuất ĐẦY ĐỦ 100% các bước hướng dẫn cài đặt/sửa lỗi tương ứng từ Kho Tri Thức ra ngay lập tức.
    2. TRÌNH BÀY RÕ RÀNG & ĐẦY ĐỦ: Đánh số thứ tự (1, 2, 3...) rõ ràng. Viết đầy đủ tất cả các bước, tuyệt đối KHÔNG cắt ngang câu giữa chừng.
    3. HÌNH ẢNH & LINK: Nếu Kho Tri Thức có link driver, link video hoặc link ảnh tem nhãn, hãy hiển thị dạng Markdown `[Tên link](URL)` hoặc `![Mô tả](URL)` để người dùng nhấp vào được.

    PHÂN QUYỀN (ROLE: {req.role}):
    - 'Khach_Hang': Hướng dẫn kỹ thuật tận tình, ngắn gọn, dễ hiểu. Nếu không xử lý được thì gợi ý liên hệ Tổng đài hỗ trợ.
    - 'Sale': Cung cấp đầy đủ SĐT Kỹ thuật + Địa chỉ bảo hành kho HN và HCM (Tab QUY_TRINH_LEO_THANG).

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

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent?key={GEMINI_API_KEY}&alt=sse"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "contents": gemini_contents,
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 2048
        }
    }

    def generate():
        try:
            res = requests.post(url, headers=headers, json=payload, stream=True, timeout=20)
            if res.status_code != 200:
                yield f"❌ Lỗi {res.status_code}: {res.text}"
                return

            for line in res.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith('data: '):
                        data_str = decoded[6:]
                        try:
                            data_json = json.loads(data_str)
                            chunk = data_json['candidates'][0]['content']['parts'][0]['text']
                            yield chunk
                        except Exception:
                            pass
        except Exception as err:
            yield f"❌ Lỗi kết nối: {str(err)}"

    return StreamingResponse(
        generate(), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )
