import os
import io
import json
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI(title="XuanTech AI Engine", version="6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SHEET_ID = "1ZMq0mTiQTDiP92UPaOIv39Q17WJXDiuvrcyYwfs7_Ag"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Cấu hình API Key tương thích 100% với cả mã AQ... lẫn AIzaSy...
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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
    Bạn là XuanTech AI (TD-Bot) – Trợ lý tư vấn & chẩn đoán sự cố thiết bị phần cứng.
    TÁC GIẢ & CHỊU TRÁCH NHIỆM: Anh Thái Đình Xuân (XuanTD) - Leader Quản lý & Phát triển thiết bị.

    QUY TẮC PHẢN HỒI:
    1. BÁM SÁT MỤC TIÊU BAN ĐẦU: Đọc kỹ lịch sử hội thoại. Nếu người dùng đang hỏi khắc phục LỖI (vd: không in được) và vừa cung cấp tên MODEL (vd: SPR02), BẮT BUỘC chỉ trả về đúng các bước khắc phục LỖI KHÔNG IN ĐƯỢC cho model SPR02 đó. Tuyệt đối KHÔNG xuất ra tài liệu cài đặt, giới thiệu hay các lỗi không liên quan.
    2. NGẮN GỌN & ĐI THẲNG VÀO ĐÁP ÁN: Trả lời trong 3-5 bước hành động chính. Không viết dài dòng giải thích.
    3. HÌNH ẢNH & ĐƯỜNG LINK: Render ảnh Markdown `![mô tả](URL)` nếu kho tri thức có link ảnh lỗi/tem. Đính kèm link driver/video nếu có.

    PHÂN QUYỀN (ROLE: {req.role}):
    - 'Khach_Hang': Khi hướng dẫn thất bại, gợi ý gọi Tổng đài `xxxx`. Tuyệt đối KHÔNG đưa SĐT/Địa chỉ riêng.
    - 'Sale': Cung cấp đầy đủ SĐT Kỹ thuật + Địa chỉ bảo hành kho HN và HCM (Tab QUY_TRINH_LEO_THANG).

    KHO TRI THỨC TRA CỨU:
    {compact_knowledge}
    """

    # Chuyển đổi lịch sử chat sang định dạng google-generativeai
    contents = []
    for m in req.messages:
        role_type = "user" if m["role"] == "user" else "model"
        contents.append({"role": role_type, "parts": [m["text"]]})

    def generate():
        try:
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=system_instruction
            )
            
            generation_config = genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=600,
            )
            
            response = model.generate_content(
                contents,
                generation_config=generation_config,
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as err:
            yield f"❌ Lỗi: {str(err)}"

    return StreamingResponse(
        generate(), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )
