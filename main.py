import os
import io
import json
import asyncio
import re
import pandas as pd
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Trợ Lý KHO Sapo Gemini Enterprise Engine", version="154.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# CẤU HÌNH BIẾN MÔI TRƯỜNG & GEMINI API
# ==========================================
SHEET_ID = os.getenv("SHEET_ID", "1ZMq0mTiQTDiP92UPaOIv39Q17WJXDiuvrcyYwfs7_Ag").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip() # Anh có thể đổi thành gemini-3.6-flash qua biến môi trường Cloud Run
SALE_SECRET_KEY = os.getenv("SALE_SECRET_KEY", "sapo2026").strip()

RAM_CACHE_SHEETS = {}
DOC_CONTENT_CACHE = {}

TABS_PUBLIC = [
    "1_THIET_BI_VA_LOI", 
    "2_HUONG_DAN_CAI_DAT", 
    "3_CHINH_SACH_SAPO", 
    "NHAN_DIEN_THIET_BI"
]
TAB_PRIVATE = "4_DU_LIEU_NOI_BO"
ALL_TABS = TABS_PUBLIC + [TAB_PRIVATE]

HTTP_CLIENT: httpx.AsyncClient = None

# TỪ ĐIỂN ĐỒNG NGHĨA KỸ THUẬT SAPO
SYNONYMS_DICT = {
    "kẹt dao": ["không cắt giấy", "lỗi cắt giấy", "kẹt dao", "hư dao cắt", "cutter"],
    "khổ giấy": ["kích thước giấy", "khổ tem", "khổ giấy in", "paper size", "kích thước tem"],
    "điện thoại": ["xtest", "app xtest", "in qua lan", "đổi ip", "android", "ios", "wifi", "không dây"],
    "máy tính": ["driver", "windows", "mac", "pc", "laptop", "cài driver", "cáp usb"],
    "in ra giấy trắng": ["không ra mực", "trắng tinh", "mờ mực", "ngược giấy"],
    "cài đặt": ["cài máy", "setup", "hướng dẫn cài", "cách cài", "cấu hình", "kết nối"]
}

# ==========================================
# KHỞI TẠO HỆ THỐNG (LIFECYCLE)
# ==========================================
# ==========================================
# KHỞI TẠO HỆ THỐNG (LIFECYCLE)
# ==========================================
@app.on_event("startup")
async def startup_event():
    global HTTP_CLIENT
    HTTP_CLIENT = httpx.AsyncClient(
        timeout=httpx.Timeout(6.0, read=8.0),
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
    )
    # 🚀 CHO PHÉP MỞ CỔNG 8080 NGAY LẬP TỨC: Tải dữ liệu chạy ngầm, không chặn Startup Probe của Cloud Run
    asyncio.create_task(load_sheet_data_async())

@app.on_event("shutdown")
async def shutdown_event():
    if HTTP_CLIENT:
        await HTTP_CLIENT.aclose()

# ==========================================
# XỬ LÝ DỮ LIỆU TỪ GOOGLE SHEET & DOCS
# ==========================================
async def fetch_single_tab_raw(tab: str):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={tab}"
    try:
        res = await HTTP_CLIENT.get(url, timeout=5.0)
        if res.status_code == 200 and "text/csv" in res.headers.get("Content-Type", ""):
            df = pd.read_csv(io.BytesIO(res.content)).fillna("")
            records = [{str(k): str(v).strip() for k, v in row.items() if str(v).strip()} for _, row in df.iterrows() if any(str(v).strip() for v in row.values)]
            return tab, records
    except Exception: pass
    return tab, []

async def fetch_google_doc_text(doc_url: str) -> str:
    if not doc_url or "docs.google.com/document" not in doc_url: return ""
    if doc_url in DOC_CONTENT_CACHE: return DOC_CONTENT_CACHE[doc_url]
    
    try:
        match = re.search(r'/d/([a-zA-Z0-9-_]+)', doc_url)
        if match:
            doc_id = match.group(1)
            export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
            res = await HTTP_CLIENT.get(export_url, timeout=4.0)
            if res.status_code == 200:
                text_content = res.text.strip()[:1800]
                DOC_CONTENT_CACHE[doc_url] = text_content
                return text_content
    except Exception: pass
    return ""

async def load_sheet_data_async():
    global RAM_CACHE_SHEETS
    tasks = [fetch_single_tab_raw(tab) for tab in ALL_TABS]
    results = await asyncio.gather(*tasks)
    RAM_CACHE_SHEETS = {tab: records for tab, records in results}
    return {"status": "success"}

@app.get("/")
def health_check():
    return {"status": "healthy", "provider": "Google Gemini", "gemini_model": GEMINI_MODEL}

@app.get("/reload")
async def reload_data():
    global DOC_CONTENT_CACHE
    DOC_CONTENT_CACHE.clear()
    return await load_sheet_data_async()

class ChatRequest(BaseModel):
    messages: list
    role: str = "Khach_Hang"

def extract_user_text(event: dict) -> str:
    if "message" in event and isinstance(event["message"], dict):
        if "text" in event["message"]: return event["message"]["text"]
        if "argumentText" in event["message"]: return event["message"]["argumentText"]
    if "chat" in event and isinstance(event["chat"], dict):
        chat = event["chat"]
        if "messagePayload" in chat and isinstance(chat["messagePayload"], dict):
            if "message" in chat["messagePayload"] and isinstance(chat["messagePayload"]["message"], dict):
                if "text" in chat["messagePayload"]["message"]: return chat["messagePayload"]["message"]["text"]
        if "message" in chat and isinstance(chat["message"], dict):
            if "text" in chat["message"]: return chat["message"]["text"]
            
    def deep_search(obj):
        if isinstance(obj, dict):
            if "argumentText" in obj and isinstance(obj["argumentText"], str) and obj["argumentText"].strip(): return obj["argumentText"]
            if "text" in obj and isinstance(obj["text"], str) and obj["text"].strip() and not obj["text"].startswith("spaces/"): return obj["text"]
            for k, v in obj.items():
                res = deep_search(v)
                if res: return res
        elif isinstance(obj, list):
            for item in obj:
                res = deep_search(item)
                if res: return res
        return ""
    return deep_search(event)

# ==========================================
# KHO TRI THỨC VÀ HYBRID INTELLIGENCE PROMPT
# ==========================================
async def get_deep_knowledge_context(query: str, role: str) -> tuple[str, list]:
    global RAM_CACHE_SHEETS
    if not RAM_CACHE_SHEETS:
        await load_sheet_data_async()

    accessible_tabs = ALL_TABS if role == "Sale" else TABS_PUBLIC
    query_lower = query.lower()

    expanded_keywords = [query_lower]
    for key, syns in SYNONYMS_DICT.items():
        if key in query_lower:
            expanded_keywords.extend(syns)

    scored_rows = []
    for tab in accessible_tabs:
        for row in RAM_CACHE_SHEETS.get(tab, []):
            row_text = " ".join(str(v).lower() for v in row.values())
            score = 0
            dev_name = str(row.get("Ten_Thiet_Bi", row.get("Loai_Thiet_Bi", ""))).lower()

            for kw in expanded_keywords:
                if len(kw) >= 2 and kw in dev_name: score += 60
                elif len(kw) >= 2 and kw in row_text: score += 5

            if score > 0:
                scored_rows.append((score, tab, row))

    scored_rows.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored_rows[:4]

    knowledge_text = ""
    raw_rows = []
    for score, tab, row in top_matches:
        raw_rows.append(row)
        knowledge_text += f"\n=== KẾT QUẢ TỪ KHO DỮ LIỆU [{tab}] ===\n"
        for key, value in row.items():
            if value:
                knowledge_text += f"- {key}: {value}\n"
                if "docs.google.com/document" in str(value):
                    doc_text = await fetch_google_doc_text(str(value))
                    if doc_text:
                        knowledge_text += f"  [NỘI DUNG TÀI LIỆU ĐÍNH KÈM]: {doc_text}\n"

    return knowledge_text, raw_rows

def build_hybrid_intelligence_prompt(knowledge_context: str) -> str:
    return f"""
    Bạn là Trợ Lý KHO Sapo – Chuyên gia IT cao cấp hỗ trợ kỹ thuật thiết bị Sapo. 

    🎯 QUY TẮC PHẢN HỒI (HYBRID INTELLIGENCE):
    1. **Tư duy liên kết:** Nếu khách hỏi chung chung (VD: "cài máy in wifi") mà không nói rõ tên máy, hãy hướng dẫn quy trình căn bản và lịch sự hỏi lại dòng máy đang dùng (SPL01, SPR02...). Nếu khách hỏi thông số, trả lời ngay và gợi ý cách thiết lập.
    2. **LUẬT THÉP CHỐNG BỊA ĐẶT:** 
       - CẤM tuyệt đối bịa ra đường link website hoặc số điện thoại hỗ trợ. 
       - Chỉ cung cấp Link nếu link đó CÓ TRONG kho dữ liệu bên dưới.
    3. Trình bày trực diện, thân thiện, gạch đầu dòng rõ ràng. Xưng "Em", gọi "Anh/chị".

    ---
    KHO DỮ LIỆU GỐC CỦA SAPO:
    {knowledge_context}
    """

def format_direct_sheet_fallback(raw_rows: list) -> str:
    if not raw_rows:
        return "👋 Dạ em là Trợ Lý KHO Sapo. Anh/chị cần hỗ trợ tra cứu thiết bị hay lỗi nào ạ?"
    lines = ["🤖 *Trợ Lý KHO Sapo (Truy xuất dữ liệu trực tiếp)*:\n"]
    for row in raw_rows[:2]:
        dev_name = row.get("Ten_Thiet_Bi", row.get("Loai_Thiet_Bi", "Thiết bị Sapo"))
        guide = row.get("Noi_Dung_Huong_Dan", row.get("Cach_Khac_Phuc", row.get("Mo_Ta_Loi", "")))
        driver = row.get("Link_Driver", row.get("Link_Video", ""))
        lines.append(f"📌 *{dev_name}*")
        if guide: lines.append(f"• Thông tin: {guide}")
        if driver: lines.append(f"• Tài liệu: {re.sub(r'\[(.*?)\]\((https?://.*?)\)', r'\\1 (\\2)', driver)}")
        lines.append("")
    return "\n".join(lines).strip()

async def call_gemini_single(system_instruction: str, user_message: str, fallback_rows: list) -> str:
    if not GEMINI_API_KEY:
        return format_direct_sheet_fallback(fallback_rows)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1000}
    }
    try:
        res = await HTTP_CLIENT.post(url, headers=headers, json=payload, timeout=6.0)
        if res.status_code == 200:
            data = res.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception: pass

    return format_direct_sheet_fallback(fallback_rows)

def wrap_gsuite_addon_response(text_message: str) -> dict:
    clean_text = re.sub(r'\[(.*?)\]\((https?://.*?)\)', r'\1 (\2)', text_message)
    return {"hostAppDataAction": {"chatDataAction": {"createMessageAction": {"message": {"text": clean_text}}}}}

# ==========================================
# CỔNG WEB VERCEL VÀ STREAMING (GEMINI STREAM)
# ==========================================
@app.post("/chat")
async def chat_stream(req: ChatRequest):
    latest_msg = req.messages[-1]["text"] if req.messages else ""
    focused_knowledge, raw_rows = await get_deep_knowledge_context(latest_msg, req.role)
    system_instruction = build_hybrid_intelligence_prompt(focused_knowledge)

    if not GEMINI_API_KEY:
        return StreamingResponse(iter([format_direct_sheet_fallback(raw_rows)]), media_type="text/plain")

    gemini_contents = []
    for m in req.messages[-5:]:
        role_type = "user" if m["role"] == "user" else "model"
        gemini_contents.append({"role": role_type, "parts": [{"text": m["text"]}]})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:streamGenerateContent?key={GEMINI_API_KEY}&alt=sse"
    headers = {"Content-Type": "application/json"}
    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": gemini_contents,
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1000}
    }

    async def generate_gemini():
        try:
            async with HTTP_CLIENT.stream("POST", url, headers=headers, json=payload, timeout=8.0) as response:
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        if line and line.startswith("data: "):
                            try:
                                data_json = json.loads(line[6:])
                                if "candidates" in data_json and data_json["candidates"]:
                                    chunk = data_json["candidates"][0]["content"]["parts"][0].get("text", "")
                                    if chunk: yield chunk
                            except Exception: pass
                    return
        except Exception: pass
        yield format_direct_sheet_fallback(raw_rows)

    return StreamingResponse(generate_gemini(), media_type="text/plain")

# ==========================================
# CỔNG GOOGLE CHAT BOT (/google-chat)
# ==========================================
@app.post("/google-chat")
async def google_chat_webhook(request: Request):
    try:
        event = await request.json()
        cleaned_message = re.sub(r'<.*?>', '', extract_user_text(event)).replace("@Trợ Lý KHO Sapo", "").strip()

        if event.get("type") == "ADDED_TO_SPACE":
            return JSONResponse(content=wrap_gsuite_addon_response("👋 Xin chào! Em là Trợ Lý KHO Sapo. Hãy gõ câu hỏi để em hỗ trợ ngay 24/7!"))

        quick_greetings = ["chào", "chào bạn", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em", "chao ban nhe"]
        if not cleaned_message or cleaned_message.lower() in quick_greetings:
            return JSONResponse(content=wrap_gsuite_addon_response("👋 Xin chào! Em là Trợ Lý KHO Sapo. Anh/chị cần hỗ trợ tra cứu thông số máy in hay cài đặt thiết bị nào ạ?"))

        focused_knowledge, raw_rows = await get_deep_knowledge_context(cleaned_message, role="Sale")
        system_instruction = build_hybrid_intelligence_prompt(focused_knowledge)

        ai_response = await call_gemini_single(system_instruction, cleaned_message, raw_rows)
        return JSONResponse(content=wrap_gsuite_addon_response(ai_response))

    except Exception:
        return JSONResponse(content=wrap_gsuite_addon_response("Dạ hệ thống đang được tải, anh/chị vui lòng thử lại câu hỏi giúp em nhé!"))
