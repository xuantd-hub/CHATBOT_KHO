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

app = FastAPI(title="Trợ Lý KHO Master Final", version="100.3")

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

async def fetch_single_tab_raw(tab: str):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={tab}"
    try:
        res = await HTTP_CLIENT.get(url, timeout=10.0)
        if res.status_code == 200 and "text/csv" in res.headers.get("Content-Type", ""):
            df = pd.read_csv(io.BytesIO(res.content)).fillna("")
            records = []
            for _, row in df.iterrows():
                row_data = {str(k): str(v).strip() for k, v in row.items() if str(v).strip()}
                if row_data:
                    records.append(row_data)
            return tab, records
    except Exception as e:
        print(f"⚠️ Lỗi tab '{tab}': {e}")
    return tab, []

async def load_sheet_data_async():
    global RAM_CACHE
    tasks = [fetch_single_tab_raw(tab) for tab in ALL_TABS]
    results = await asyncio.gather(*tasks)
    RAM_CACHE = {tab: records for tab, records in results}
    print("✅ Đã nạp 100% dữ liệu thô, giữ nguyên chi tiết!")
    return {"status": "success"}

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

def get_smart_focused_knowledge(query: str, role: str) -> str:
    stop_words = {"mình", "có", "bị", "được", "không", "cho", "với", "là", "và", "nhé", "ạ", "cần", "giúp", "tôi", "xin", "lỗi", "máy", "thế", "nào", "bao", "nhiêu"}
    words = [w.lower() for w in query.split() if len(w) > 1 and w.lower() not in stop_words]
    if not words:
        words = [query.lower()]

    accessible_tabs = ALL_TABS if role == "Sale" else TABS_PUBLIC
    scored_rows = []

    for tab in accessible_tabs:
        for row in RAM_CACHE.get(tab, []):
            row_text = " ".join(str(v).lower() for v in row.values())
            
            score = 0
            for w in words:
                if w in row_text:
                    score += 1
            
            model_name = str(row.get("Ten_Thiet_Bi", "")).lower()
            if any(w in model_name for w in words):
                score += 10

            if score > 0:
                scored_rows.append((score, tab, row))

    scored_rows.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored_rows[:8]

    if not top_matches:
        return "Không tìm thấy dữ liệu khớp lệnh. Dựa vào kiến thức sẵn có, hãy tư vấn nhẹ nhàng."

    knowledge_text = ""
    for score, tab, row in top_matches:
        knowledge_text += f"\n--- [Nguồn: {tab}] ---\n"
        for key, value in row.items():
            knowledge_text += f"{key}: {value}\n"
    
    return knowledge_text

@app.post("/chat")
async def chat_stream(req: ChatRequest):
    latest_msg = req.messages[-1]["text"] if req.messages else ""
    focused_knowledge = get_smart_focused_knowledge(latest_msg, req.role)

    system_instruction = f"""
    Bạn là Trợ Lý KHO – Chuyên gia tư vấn & kỹ thuật phần cứng của Sapo. Thông minh, tận tâm và chuyên nghiệp.

    NHIỆM VỤ CỦA BẠN:
    1. Đọc thật kỹ các "Khối Thông Tin" được trích xuất từ Kho Tri Thức dưới đây.
    2. TẬN DỤNG TỐI ĐA SỰ THÔNG MINH: Giải thích rõ ràng, rành mạch từng bước (Bước 1, Bước 2...) cho khách dễ hiểu. KHÔNG được cộc lốc chỉ quăng mỗi link.
    3. Đính kèm đầy đủ link tải Driver/Video bằng Markdown `[Tên](Link)` đan xen vào các bước hoặc để ở cuối.
    4. CHÍNH XÁC TUYỆT ĐỐI (ZERO HALLUCINATION): Chỉ cung cấp thông số, địa chỉ, SĐT nếu có trong Kho Tri Thức. Nếu dữ liệu không có thông tin khách hỏi, hãy nhẹ nhàng báo chưa cập nhật và hướng dẫn liên hệ Tổng đài Sapo. TUYỆT ĐỐI KHÔNG BỊA ĐẶT.
    5. ĐỊNH DẠNG: Dùng `➔` để chỉ hướng. Tên bạn là "Trợ Lý KHO".

    BẢO MẬT (ROLE: {req.role}):
    - Nếu là Khách Hàng: Dữ liệu nhạy cảm đã bị hệ thống chặn, cứ tư vấn bình thường.
    - Nếu là Sale: Bạn sẽ thấy có thông tin nội bộ (Tab 4), hãy cung cấp đầy đủ cho Sale.

    KHO TRI THỨC ĐƯỢC TRÍCH XUẤT CHO CÂU HỎI NÀY:
    {focused_knowledge}
    """

    trimmed_messages = req.messages[-5:] if len(req.messages) > 5 else req.messages
    gemini_contents = []
    for m in trimmed_messages:
        role_type = "user" if m["role"] == "user" else "model"
        gemini_contents.append({
            "role": role_type,
            "parts": [{"text": m["text"]}]
        })

    # Sử dụng đúng model 3.6 flash siêu tốc như đã thống nhất
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
        has_yielded = False
        try:
            async with HTTP_CLIENT.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    err_body = await response.aread()
                    yield f"❌ Lỗi API Google ({response.status_code}): {err_body.decode('utf-8')}"
                    return

                async for line in response.aiter_lines():
                    if line and line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            data_json = json.loads(data_str)
                            if "candidates" in data_json and len(data_json["candidates"]) > 0:
                                candidate = data_json["candidates"][0]
                                if "content" in candidate and "parts" in candidate["content"]:
                                    chunk = candidate["content"]["parts"][0].get("text", "")
                                    if chunk:
                                        has_yielded = True
                                        yield chunk
                                elif "finishReason" in candidate and candidate["finishReason"] != "STOP":
                                    has_yielded = True
                                    yield f"\n\n*(Hệ thống ngừng xuất chữ do: {candidate['finishReason']})*"
                        except Exception:
                            pass
        except httpx.ReadTimeout:
            yield "❌ Lỗi: Máy chủ Google AI phản hồi quá lâu (Timeout). Vui lòng thử lại."
            return
        except Exception as err:
            yield f"❌ Lỗi kết nối: {str(err)}"
            return
            
        if not has_yielded:
            yield "❌ Lỗi: Máy chủ Google AI trả về phản hồi rỗng (Có thể do lỗi định dạng hoặc từ khóa bị chặn)."

    return StreamingResponse(
        generate(), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
