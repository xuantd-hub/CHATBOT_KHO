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

app = FastAPI(title="Trợ Lý KHO Engine Pro", version="21.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SHEET_ID = "1ZMq0mTiQTDiP92UPaOIv39Q17WJXDiuvrcyYwfs7_Ag"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
SALE_SECRET_KEY = os.getenv("SALE_SECRET_KEY", "sapo2026").strip()

RAM_CACHE = {}

# Cấu trúc 5 Tab chốt chuẩn xác
TABS_PUBLIC = [
    "1_THIET_BI_VA_LOI", 
    "2_HUONG_DAN_CAI_DAT", 
    "3_CHINH_SACH_SAPO", 
    "NHAN_DIEN_THIET_BI"
]
TAB_PRIVATE = "4_DU_LIEU_NOI_BO"
ALL_TABS = TABS_PUBLIC + [TAB_PRIVATE]

# Duy trì HTTP Client dùng chung để tối ưu tốc độ kết nối SSL
HTTP_CLIENT: httpx.AsyncClient = None

@app.on_event("startup")
async def startup_event():
    global HTTP_CLIENT
    HTTP_CLIENT = httpx.AsyncClient(timeout=httpx.Timeout(15.0, read=30.0))
    await load_sheet_data_async()

@app.on_event("shutdown")
async def shutdown_event():
    if HTTP_CLIENT:
        await HTTP_CLIENT.aclose()

async def fetch_single_tab(tab: str):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={tab}"
    try:
        res = await HTTP_CLIENT.get(url, timeout=5.0)
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
    tasks = [fetch_single_tab(tab) for tab in ALL_TABS]
    results = await asyncio.gather(*tasks)
    RAM_CACHE = {tab: records for tab, records in results}
    print("✅ Đã nạp thành công dữ liệu 5 Tab vào RAM!")
    return {"status": "success", "loaded_tabs": list(RAM_CACHE.keys())}

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
    else:
        return {"success": False, "message": "Mật khẩu nội bộ chưa chính xác!"}

class ChatRequest(BaseModel):
    messages: list
    role: str = "Khach_Hang"

def filter_relevant_knowledge(latest_user_msg: str, role: str) -> str:
    # Bộ lọc từ dừng tiếng Việt giúp loại bỏ từ thừa, tập trung vào Keyword chính
    stop_words = {
        "mình", "có", "bị", "được", "không", "cho", "với", "là", "và", "nhé", 
        "ạ", "cần", "giúp", "hướng", "dẫn", "tôi", "cho", "xin", "lỗi", "máy"
    }
    raw_words = [w.lower() for w in latest_user_msg.split() if len(w) > 1]
    query_words = [w for w in raw_words if w not in stop_words]
    if not query_words:
        query_words = raw_words

    filtered_data = {}
    accessible_tabs = ALL_TABS if role == "Sale" else TABS_PUBLIC

    for tab_name in accessible_tabs:
        rows = RAM_CACHE.get(tab_name, [])
        matched_rows = []
        for row in rows:
            row_text = " ".join(row.values()).lower()
            if any(word in row_text for word in query_words):
                matched_rows.append(row)
        if matched_rows:
            filtered_data[tab_name] = matched_rows[:2]  # Lấy tối đa 2 dòng khớp nhất để Prompt siêu nhẹ

    if not filtered_data:
        filtered_data = {
            "3_CHINH_SACH_SAPO": RAM_CACHE.get("3_CHINH_SACH_SAPO", [])[:1],
            "2_HUONG_DAN_CAI_DAT": RAM_CACHE.get("2_HUONG_DAN_CAI_DAT", [])[:1],
            "NHAN_DIEN_THIET_BI": RAM_CACHE.get("NHAN_DIEN_THIET_BI", [])[:1]
        }
        if role == "Sale":
            filtered_data["4_DU_LIEU_NOI_BO"] = RAM_CACHE.get("4_DU_LIEU_NOI_BO", [])[:1]

    compact_str = json.dumps(filtered_data, ensure_ascii=False, separators=(',', ':'))
    return compact_str[:1200]  # Giới hạn dung lượng dữ liệu tối đa 1200 ký tự

@app.post("/chat")
async def chat_stream(req: ChatRequest):
    latest_msg = req.messages[-1]["text"] if req.messages else ""
    compact_knowledge = filter_relevant_knowledge(latest_msg, req.role)

    system_instruction = f"""
    Bạn là Trợ Lý KHO – Trợ lý tư vấn & chẩn đoán sự cố thiết bị phần cứng Sapo.
    WEBSITE: https://sapo.vn | THIẾT BỊ: https://shop.sapo.vn

    QUY TẮC PHẢN HỒI NỘI DUNG:
    1. TRÍCH XUẤT ĐẦY ĐỦ CHI TIẾT TỪNG BƯỚC:
       - Hướng dẫn cài đặt, chẩn đoán lỗi, chính sách bảo hành: BẮT BUỘC xuất đầy đủ các bước thao tác kỹ thuật có trong Kho Tri Thức.
       - Viết trọn vẹn cú pháp Markdown link `[Tên hiển thị](URL)`. Không bỏ dở link.
    2. NGUYÊN TẮC CHỐNG BỊA ĐỊA CHỈ:
       - Chỉ cung cấp địa chỉ/SĐT có trong Kho Tri Thức. Nếu thiếu, hướng dẫn truy cập https://shop.sapo.vn hoặc Tổng đài Sapo.
    3. BẢO MẬT NỘI BỘ (ROLE HIỆN TẠI: {req.role}):
       - Role 'Khach_Hang': Tuyệt đối KHÔNG tiết lộ SĐT kỹ thuật trực ca, chiết khấu hay địa chỉ kho riêng.
       - Role 'Sale': Mới mở khóa dữ liệu bảo hành nội bộ.
    4. TRÌNH BÀY: Dùng ký tự `➔` hoặc `->` khi hướng dẫn bấm menu. KHÔNG dùng LaTeX. Tên bạn là "Trợ Lý KHO". Chỉ khi người dùng hỏi "Ai tạo ra bạn?" mới nêu tên tác giả Thái Đình Xuân (XuanTD) - Nhân viên Quản lý & Phát triển thiết bị.

    KHO TRI THỨC TRA CỨU:
    {compact_knowledge}
    """

    trimmed_messages = req.messages[-4:] if len(req.messages) > 4 else req.messages
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
            "maxOutputTokens": 2048
        }
    }

    async def generate():
        try:
            async with HTTP_CLIENT.stream("POST", url, headers=headers, json=payload) as response:
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
        except Exception as err:
            yield f"❌ Lỗi kết nối: {str(err)}"

    return StreamingResponse(
        generate(), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
