import os
import io
import asyncio
import pandas as pd
import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Trợ Lý KHO Engine Master Uncut", version="27.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SHEET_ID = "1ZMq0mTiQTDiP92UPaOIv39Q17WJXDiuvrcyYwfs7_Ag"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
SALE_SECRET_KEY = os.getenv("SALE_SECRET_KEY", "sapo2026").strip()

# Bộ nhớ RAM lưu trữ văn bản Markdown nguyên bản 100% của từng Tab
RAM_MARKDOWN_CACHE = {}

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
    HTTP_CLIENT = httpx.AsyncClient(timeout=httpx.Timeout(20.0, read=40.0))
    await load_sheet_data_async()

@app.on_event("shutdown")
async def shutdown_event():
    if HTTP_CLIENT:
        await HTTP_CLIENT.aclose()

async def fetch_single_tab_markdown(tab: str):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={tab}"
    try:
        res = await HTTP_CLIENT.get(url, timeout=6.0)
        if res.status_code == 200 and "text/csv" in res.headers.get("Content-Type", ""):
            df = pd.read_csv(io.BytesIO(res.content)).fillna("")
            
            # Chuyển đổi toàn bộ Tab Google Sheet thành bảng Markdown siêu gọn nhẹ
            if not df.empty:
                md_text = f"### TAB: {tab}\n"
                md_text += df.to_markdown(index=False) + "\n\n"
                return tab, md_text
    except Exception as e:
        print(f"⚠️ Cảnh báo tab '{tab}': {e}")
    return tab, f"### TAB: {tab}\n(Trống)\n\n"

async def load_sheet_data_async():
    global RAM_MARKDOWN_CACHE
    tasks = [fetch_single_tab_markdown(tab) for tab in ALL_TABS]
    results = await asyncio.gather(*tasks)
    RAM_MARKDOWN_CACHE = {tab: md_content for tab, md_content in results}
    print("✅ Đã nạp TRỌN VẸN 100% dữ liệu Google Sheet dưới dạng Markdown!")
    return {"status": "success", "loaded_tabs": list(RAM_MARKDOWN_CACHE.keys())}

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

def get_100_percent_uncut_knowledge(role: str) -> str:
    """
    TRUYỀN 100% KHÔNG RÚT GỌN TOÀN BỘ FILE GOOGLE SHEET SANG AI:
    - Khách hàng: Nhận 100% nội dung 4 Tab công khai.
    - Sale: Nhận 100% nội dung cả 5 Tab (bao gồm Tab Nội Bị).
    Không lọc bỏ, không cắt bớt bất kỳ dòng nào!
    """
    accessible_tabs = ALL_TABS if role == "Sale" else TABS_PUBLIC
    full_md = ""
    for tab in accessible_tabs:
        full_md += RAM_MARKDOWN_CACHE.get(tab, "")
    return full_md

@app.post("/chat")
async def chat_stream(req: ChatRequest):
    # Lấy trọn vẹn 100% văn bản Markdown của các Tab được phép truy cập
    uncut_knowledge_context = get_100_percent_uncut_knowledge(req.role)

    system_instruction = f"""
    Bạn là Trợ Lý KHO – Trợ lý chuyên gia tư vấn & chẩn đoán sự cố thiết bị phần cứng Sapo.
    WEBSITE CHÍNH THỨC: https://sapo.vn | THIẾT BỊ: https://shop.sapo.vn

    QUY TẮC PHẢN HỒI NỘI DUNG (CHÍNH XÁC TUYỆT ĐỐI 100% - CHỐNG MƠ HỒ):
    1. TRÍCH XUẤT ĐẦY ĐỦ VĂN BẢN VÀ LINK DRIVER TỪ KHO TRI THỨC:
       - Rà soát TOÀN BỘ bảng dữ liệu Markdown trong Kho Tri Thức dưới đây.
       - Khi trả lời về bất kỳ thiết bị nào (G8, K200L, SPR02, SPL01...): BẮT BUỘC trích xuất ĐẦY ĐỦ 100% các đường link Drive trong các cột 'Link_Driver_Win', 'Link_Driver_Mac', 'Link_Video_Huong_Dan' hoặc nội dung hướng dẫn đi kèm.
       - Viết trọn vẹn cú pháp Markdown link `[Tên hiển thị](URL)`. TUYỆT ĐỐI KHÔNG tự ý đẩy khách ra trang web ngoài khi Kho Tri Thức đã có link Drive cụ thể.
    2. CHỐNG BỊA ĐỊA CHỈ: Chỉ cung cấp địa chỉ kho/SĐT bảo hành khi thông tin đó CÓ TRONG Kho Tri Thức. Nếu thiếu, hướng dẫn liên hệ Tổng đài Sapo.
    3. BẢO MẬT DỮ LIỆU NỘI BỘ (QUYỀN HIỆN TẠI: {req.role}):
       - NẾU ROLE 'Khach_Hang': Tuyệt đối không tiết lộ thông tin nội bộ (Tab 4 đã được tự động khóa hoàn toàn).
       - NẾU ROLE 'Sale': Mới cung cấp quy trình bảo hành nội bộ, SĐT kỹ thuật trực ca và ghi chú ưu đãi.
    4. CHUẨN ĐỊNH DẠNG:
       - Dùng ký tự Unicode `➔` hoặc `->` khi hướng dẫn bấm menu. TUYỆT ĐỐI KHÔNG dùng LaTeX (`$\\rightarrow$`).
       - Xưng danh: "Trợ Lý KHO". Chỉ khi người dùng hỏi "Ai tạo ra bạn?" mới trả lời: "Hệ thống được phát triển bởi anh Thái Đình Xuân (XuanTD) - Nhân viên Quản lý & Phát triển thiết bị."

    KHO TRI THỨC TOÀN DIỆN (ĐÃ NẠP 100% DỮ LIỆU GOOGLE SHEET DẠNG MARKDOWN):
    {uncut_knowledge_context}
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
            "temperature": 0.0,
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
