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

app = FastAPI(title="Trợ Lý KHO Engine", version="19.0")

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

# Cấu trúc 5 Tab chốt theo yêu cầu thực tế
TABS_PUBLIC = [
    "1_THIET_BI_VA_LOI", 
    "2_HUONG_DAN_CAI_DAT", 
    "3_CHINH_SACH_SAPO", 
    "NHAN_DIEN_THIET_BI"
]
TAB_PRIVATE = "4_DU_LIEU_NOI_BO"
ALL_TABS = TABS_PUBLIC + [TAB_PRIVATE]

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
        tasks = [fetch_single_tab(client, tab) for tab in ALL_TABS]
        results = await asyncio.gather(*tasks)
        
    RAM_CACHE = {tab: records for tab, records in results}
    print("✅ Đã nạp thành công 5 Tab Google Sheet vào RAM!")
    return {"status": "success", "loaded_tabs": list(RAM_CACHE.keys())}

@app.on_event("startup")
async def startup_event():
    await load_sheet_data_async()

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
    query_words = [w.lower() for w in latest_user_msg.split() if len(w) > 1]
    filtered_data = {}

    # Khách hàng chỉ truy cập TABS_PUBLIC. Sale được mở rộng thêm TAB_PRIVATE.
    accessible_tabs = ALL_TABS if role == "Sale" else TABS_PUBLIC

    for tab_name in accessible_tabs:
        rows = RAM_CACHE.get(tab_name, [])
        matched_rows = []
        for row in rows:
            row_text = " ".join(row.values()).lower()
            if any(word in row_text for word in query_words):
                matched_rows.append(row)
        if matched_rows:
            filtered_data[tab_name] = matched_rows[:3]

    # Nếu không khớp từ khóa cụ thể, trả về dữ liệu mặc định an toàn
    if not filtered_data:
        filtered_data = {
            "3_CHINH_SACH_SAPO": RAM_CACHE.get("3_CHINH_SACH_SAPO", [])[:2],
            "2_HUONG_DAN_CAI_DAT": RAM_CACHE.get("2_HUONG_DAN_CAI_DAT", [])[:2],
            "NHAN_DIEN_THIET_BI": RAM_CACHE.get("NHAN_DIEN_THIET_BI", [])[:2]
        }
        if role == "Sale":
            filtered_data["4_DU_LIEU_NOI_BO"] = RAM_CACHE.get("4_DU_LIEU_NOI_BO", [])[:2]

    return json.dumps(filtered_data, ensure_ascii=False, separators=(',', ':'))

@app.post("/chat")
async def chat_stream(req: ChatRequest):
    latest_msg = req.messages[-1]["text"] if req.messages else ""
    compact_knowledge = filter_relevant_knowledge(latest_msg, req.role)

    system_instruction = f"""
    Bạn là Trợ Lý KHO – Trợ lý tư vấn & chẩn đoán sự cố thiết bị phần cứng chuyên nghiệp của Sapo.

    WEBSITE THAM CHIẾU CHÍNH THỨC:
    - Trang chủ chính: https://sapo.vn
    - Trang thiết bị phần cứng: https://shop.sapo.vn

    QUY TẮC PHẢN HỒI NỘI DUNG (CHÍNH XÁC & BẢO MẬT KHẮC KHET):
    1. TRÍCH XUẤT 100% NỘI DUNG CHI TIẾT TỪNG BƯỚC:
       - Khi trả lời về hướng dẫn cài đặt, chẩn đoán lỗi hay chính sách bảo hành: BẮT BUỘC trích xuất ĐẦY ĐỦ chi tiết từng bước thao tác trong Kho Tri Thức (nhấn nút nào, giữ bao nhiêu giây, cổng USB/LAN, khổ giấy 80mm...).
       - Viết trọn vẹn cú pháp Markdown link `[Tên hiển thị](URL)`. TUYỆT ĐỐI KHÔNG ngắt dở dở link hay tóm tắt làm mất thông tin kỹ thuật.
    2. CHỐNG BỊA ĐỊA CHỈ / THÔNG TIN BẢO HÀNH:
       - CHỈ CUNG CẤP địa chỉ bảo hành, SĐT khi thông tin đó CÓ TRONG KHO TRI THỨC được cấp.
       - Nếu thông tin chưa có trong Kho Tri Thức, báo rõ chưa cập nhật và hướng dẫn truy cập https://shop.sapo.vn hoặc liên hệ Tổng đài Sapo.
    3. NGUYÊN TẮC BẢO MẬT DỮ LIỆU NỘI BỘ (QUAN TRỌNG):
       - Hiện tại phân quyền đang là ROLE: {req.role}.
       - NẾU ROLE LA 'Khach_Hang': TUYỆT ĐỐI KHÔNG tiết lộ bất kỳ thông tin nhạy cảm nội bộ nào (SĐT kỹ thuật trực ca, chiết khấu nội bộ, địa chỉ kho riêng). Chỉ hướng dẫn kỹ thuật công khai và chính sách chung.
       - NẾU ROLE LÀ 'Sale': Mới cung cấp đầy đủ quy trình bảo hành nội bộ, SĐT Kỹ thuật và ghi chú bảo mật.
    4. QUY TẮC TRÌNH BÀY & TÁC GIẢ:
       - TUYỆT ĐỐI KHÔNG DÙNG LATEX (`$\\rightarrow$`, `\\rightarrow`, `\\$`). Dùng ký tự Unicode `➔` hoặc `->` khi hướng dẫn bấm menu.
       - Tên của bạn là "Trợ Lý KHO". Không tự chèn tên tác giả vào câu chào.
       - Chỉ khi người dùng hỏi "Ai tạo ra bạn?" mới nêu tên tác giả Thái Đình Xuân (XuanTD) - Nhân viên Quản lý & Phát triển thiết bị.

    KHO TRI THỨC TRA CỨU:
    {compact_knowledge}
    """

    trimmed_messages = req.messages[-6:] if len(req.messages) > 6 else req.messages
    gemini_contents = []
    for m in trimmed_messages:
        role_type = "user" if m["role"] == "user" else "model"
        gemini_contents.append({
            "role": role_type,
            "parts": [{"text": m["text"]}]
        })

    # Sử dụng chuẩn model mới nhất gemini-3.6-flash
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
