import os
import io
import asyncio
import json
import pandas as pd
import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Trợ Lý KHO Master Final", version="99.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SHEET_ID = "1ZMq0mTiQTDiP92UPaOIv39Q17WJXDiuvrcyYwfs7_Ag"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
SALE_SECRET_KEY = os.getenv("SALE_SECRET_KEY", "sapo2026").strip()

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
    # Keep-alive connection để truy xuất AI không độ trễ
    HTTP_CLIENT = httpx.AsyncClient(timeout=httpx.Timeout(20.0, read=40.0))
    await load_sheet_data_async()

@app.on_event("shutdown")
async def shutdown_event():
    if HTTP_CLIENT:
        await HTTP_CLIENT.aclose()

async def fetch_single_tab_markdown(tab: str):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={tab}"
    try:
        res = await HTTP_CLIENT.get(url, timeout=10.0)
        if res.status_code == 200 and "text/csv" in res.headers.get("Content-Type", ""):
            df = pd.read_csv(io.BytesIO(res.content)).fillna("")
            if not df.empty:
                # Dựng bảng Markdown chuẩn xác 100% bằng Python thuần (chống crash thư viện)
                headers_str = "| " + " | ".join(df.columns.astype(str)) + " |"
                separator_str = "| " + " | ".join(["---"] * len(df.columns)) + " |"
                
                lines = [f"### BẢNG DỮ LIỆU TAB: {tab}", headers_str, separator_str]
                for _, row in df.iterrows():
                    # Xóa ký tự xuống dòng trong ô để không làm vỡ cấu trúc bảng
                    row_vals = [str(v).strip().replace('\n', ' ') for v in row.values]
                    lines.append("| " + " | ".join(row_vals) + " |")
                
                return tab, "\n".join(lines) + "\n\n"
    except Exception as e:
        print(f"⚠️ Lỗi tab '{tab}': {e}")
    return tab, f"### BẢNG DỮ LIỆU TAB: {tab}\n(Trống)\n\n"

async def load_sheet_data_async():
    global RAM_MARKDOWN_CACHE
    tasks = [fetch_single_tab_markdown(tab) for tab in ALL_TABS]
    results = await asyncio.gather(*tasks)
    RAM_MARKDOWN_CACHE = {tab: md_content for tab, md_content in results}
    print("✅ Đã nạp thành công 100% dữ liệu Google Sheet!")
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

def get_100_percent_knowledge(role: str) -> str:
    """ Truyền 100% dữ liệu. Khách hàng bị chặn hoàn toàn Tab 4 ở cấp độ Server """
    accessible_tabs = ALL_TABS if role == "Sale" else TABS_PUBLIC
    full_md = ""
    for tab in accessible_tabs:
        full_md += RAM_MARKDOWN_CACHE.get(tab, "")
    return full_md

@app.post("/chat")
async def chat_stream(req: ChatRequest):
    # Nạp nguyên bản 100% dữ liệu
    knowledge_context = get_100_percent_knowledge(req.role)

    system_instruction = f"""
    Bạn là Trợ Lý KHO – Trợ lý chuyên gia kỹ thuật và chẩn đoán phần cứng của Sapo.

    LỆNH VẬN HÀNH TUYỆT ĐỐI (PHẢI TUÂN THỦ 100%):
    1. ĐỘ CHÍNH XÁC & TRÍCH XUẤT DỮ LIỆU:
       - Bạn được cung cấp Bảng Dữ Liệu bên dưới. TẤT CẢ câu trả lời phải lấy trực tiếp từ Bảng Dữ Liệu này.
       - Khi trả lời hướng dẫn cài đặt hoặc xử lý lỗi: BẮT BUỘC liệt kê đầy đủ 100% các bước kỹ thuật chi tiết. KHÔNG tự ý tóm tắt, KHÔNG rút gọn.
       - BẮT BUỘC xuất chính xác các đường link Driver/Video bằng cú pháp Markdown `[Tên hiển thị](URL)`. KHÔNG bao giờ được bỏ dở link.
    2. CHỐNG BỊA ĐẶT (ZERO HALLUCINATION):
       - TUYỆT ĐỐI KHÔNG tự bịa ra địa chỉ, số điện thoại, hay chính sách bảo hành nếu nó không có trong Bảng Dữ Liệu.
       - Nếu dữ liệu không có, phản hồi: "Dữ liệu hiện tại chưa cập nhật thông tin này, vui lòng tham khảo https://shop.sapo.vn hoặc liên hệ Tổng đài Sapo."
    3. BẢO MẬT & PHÂN QUYỀN (ROLE: {req.role}):
       - Trạng thái Khách Hàng: KHÔNG ĐƯỢC tiết lộ thông tin nội bộ (bởi vì dữ liệu nội bộ đã bị ẩn).
       - Trạng thái Sale: Cung cấp đầy đủ thông tin bảo hành, SĐT kỹ thuật nội bộ, ghi chú.
    4. TRÌNH BÀY:
       - Xưng danh là "Trợ Lý KHO".
       - Dùng ký tự `➔` để hướng dẫn bấm menu. KHÔNG dùng LaTeX.

    BẢNG DỮ LIỆU KHO TRI THỨC (TRỌN VẸN 100%):
    {knowledge_context}
    """

    trimmed_messages = req.messages[-4:] if len(req.messages) > 4 else req.messages
    gemini_contents = []
    for m in trimmed_messages:
        role_type = "user" if m["role"] == "user" else "model"
        gemini_contents.append({
            "role": role_type,
            "parts": [{"text": m["text"]}]
        })

    # Cố định model 3.6 flash siêu tốc
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:streamGenerateContent?key={GEMINI_API_KEY}&alt=sse"
    headers = {"Content-Type": "application/json"}
    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": gemini_contents,
        "generationConfig": {
            "temperature": 0.0, # 0.0 đảm bảo không bao giờ bịa chữ, chính xác tuyệt đối với Sheet
            "maxOutputTokens": 4096 # Cho phép trả lời bài siêu dài nếu cần
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
