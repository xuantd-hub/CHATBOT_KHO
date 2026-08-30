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

app = FastAPI(title="Trợ Lý KHO Engine Master", version="25.0")

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

# Cấu trúc 5 Tab chuẩn hóa chính thức
TABS_PUBLIC = [
    "1_THIET_BI_VA_LOI", 
    "2_HUONG_DAN_CAI_DAT", 
    "3_CHINH_SACH_SAPO", 
    "NHAN_DIEN_THIET_BI"
]
TAB_PRIVATE = "4_DU_LIEU_NOI_BO"
ALL_TABS = TABS_PUBLIC + [TAB_PRIVATE]

# Duy trì kết nối SSL thường trực siêu tốc
HTTP_CLIENT: httpx.AsyncClient = None

@app.on_event("startup")
async def startup_event():
    global HTTP_CLIENT
    HTTP_CLIENT = httpx.AsyncClient(timeout=httpx.Timeout(20.0, read=40.0))
    await load_sheet_data_async()

@app.on_event("shutdown")
async def shutdown_event():
    if HTTP_CLIENT:
        await HTTP_CLIENT.aclose()

async def fetch_single_tab(tab: str):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={tab}"
    try:
        res = await HTTP_CLIENT.get(url, timeout=6.0)
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
    print("✅ Đã nạp thành công TRỌN VẸN 5 Tab Google Sheet vào RAM!")
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

def get_full_accessible_knowledge(role: str) -> str:
    """
    Nạp 100% DỮ LIỆU NGUYÊN BẢN từ Google Sheet cho AI đọc trực tiếp.
    Khách hàng: Nạp 100% dữ liệu 4 Tab công khai.
    Sale: Nạp 100% dữ liệu cả 5 Tab (bao gồm Tab Nội Bị).
    Không cắt xén, không bỏ sót bất kỳ dòng nào!
    """
    accessible_tabs = ALL_TABS if role == "Sale" else TABS_PUBLIC
    full_knowledge = {}
    for tab in accessible_tabs:
        full_knowledge[tab] = RAM_CACHE.get(tab, [])
    
    return json.dumps(full_knowledge, ensure_ascii=False, separators=(',', ':'))

@app.post("/chat")
async def chat_stream(req: ChatRequest):
    # Nạp toàn bộ dữ liệu Google Sheet tương ứng với quyền truy cập
    full_knowledge_context = get_full_accessible_knowledge(req.role)

    system_instruction = f"""
    Bạn là Trợ Lý KHO – Trợ lý chuyên gia tư vấn & chẩn đoán sự cố thiết bị phần cứng Sapo.
    WEBSITE CHÍNH THỨC: https://sapo.vn | THIẾT BỊ: https://shop.sapo.vn

    QUY TẮC PHẢN HỒI NỘI DUNG (CHÍNH XÁC TUYỆT ĐỐI 100% - CHỐNG MƠ HỒ):
    1. TRÍCH XUẤT CHÍNH XÁC ĐẦY ĐỦ VĂN BẢN VÀ LINK DRIVER TỪ KHO TRI THỨC:
       - Rà soát TOÀN BỘ dữ liệu trong Kho Tri Thức được cấp dưới đây.
       - Khi người dùng hỏi về bất kỳ thiết bị nào (ví dụ: G8, K200L, SPR02, SPL01...): BẮT BUỘC trích xuất ĐẦY ĐỦ 100% các đường link Drive trong cột 'Link_Driver_Win', 'Link_Driver_Mac', 'Link_Video_Huong_Dan' hoặc nội dung hướng dẫn thao tác đi kèm.
       - Viết trọn vẹn cú pháp Markdown link `[Tên hiển thị](URL)`. TUYỆT ĐỐI KHÔNG tự ý chuyển hướng khách sang trang web chung chung khi Kho Tri Thức đã có link Drive cụ thể.
    2. NGUYÊN TẮC THỰC TẾ & CHỐNG BỊA ĐỊA CHỈ:
       - Chỉ cung cấp địa chỉ kho/SĐT bảo hành khi thông tin đó CÓ TRONG Kho Tri Thức. Nếu dữ liệu chưa có, báo rõ chưa cập nhật và hướng dẫn liên hệ Tổng đài Sapo.
    3. BẢO MẬT DỮ LIỆU NỘI BỘ (QUYỀN HIỆN TẠI: {req.role}):
       - NẾU ROLE 'Khach_Hang': Tuyệt đối không tiết lộ thông tin nội bộ (Tab 4 đã được tự động khóa hoàn toàn).
       - NẾU ROLE 'Sale': Mới cung cấp quy trình bảo hành nội bộ, SĐT kỹ thuật trực ca và ghi chú ưu đãi.
    4. CHUẨN ĐỊNH DẠNG:
       - Dùng ký tự Unicode `➔` hoặc `->` khi hướng dẫn bấm menu. TUYỆT ĐỐI KHÔNG dùng LaTeX (`$\\rightarrow$`).
       - Xưng danh: "Trợ Lý KHO". Chỉ khi người dùng hỏi "Ai tạo ra bạn?" mới trả lời: "Hệ thống được phát triển bởi anh Thái Đình Xuân (XuanTD) - Nhân viên Quản lý & Phát triển thiết bị."

    KHO TRI THỨC TOÀN DIỆN (ĐÃ NẠP 100% DỮ LIỆU GOOGLE SHEET):
    {full_knowledge_context}
    """

    # Giữ 6 câu thoại gần nhất để AI hiểu ngữ cảnh trò chuyện
    trimmed_messages = req.messages[-6:] if len(req.messages) > 6 else req.messages
    gemini_contents = []
    for m in trimmed_messages:
        role_type = "user" if m["role"] == "user" else "model"
        gemini_contents.append({
            "role": role_type,
            "parts": [{"text": m["text"]}]
        })

    # Cố định model đỉnh cao gemini-3.6-flash
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:streamGenerateContent?key={GEMINI_API_KEY}&alt=sse"
    headers = {"Content-Type": "application/json"}
    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": gemini_contents,
        "generationConfig": {
            "temperature": 0.0,  # Hạ nhiệt độ về 0.0 để AI đạt độ chính xác tuyệt đối, trung thực 100% với file Sheet
            "maxOutputTokens": 3000
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
