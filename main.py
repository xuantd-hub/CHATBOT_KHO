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
from google import genai
from google.genai import types

app = FastAPI(title="Trợ Lý KHO Sapo GenAI SDK Engine", version="141.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SHEET_ID = os.getenv("SHEET_ID", "1ZMq0mTiQTDiP92UPaOIv39Q17WJXDiuvrcyYwfs7_Ag").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
SALE_SECRET_KEY = os.getenv("SALE_SECRET_KEY", "sapo2026").strip()

# Khởi tạo GenAI Client chính thức (hỗ trợ hoàn hảo khóa AQ. mới nhất)
genai_client = None
if GEMINI_API_KEY:
    try:
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        pass

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
    HTTP_CLIENT = httpx.AsyncClient(
        timeout=httpx.Timeout(12.0, read=15.0),
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
    )
    asyncio.create_task(load_sheet_data_async())

@app.on_event("shutdown")
async def shutdown_event():
    if HTTP_CLIENT:
        await HTTP_CLIENT.aclose()

async def fetch_single_tab_raw(tab: str):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={tab}"
    try:
        res = await HTTP_CLIENT.get(url, timeout=6.0)
        if res.status_code == 200 and "text/csv" in res.headers.get("Content-Type", ""):
            df = pd.read_csv(io.BytesIO(res.content)).fillna("")
            records = []
            for _, row in df.iterrows():
                row_data = {str(k): str(v).strip() for k, v in row.items() if str(v).strip()}
                if row_data:
                    records.append(row_data)
            return tab, records
    except Exception: pass
    return tab, []

async def load_sheet_data_async():
    global RAM_CACHE
    tasks = [fetch_single_tab_raw(tab) for tab in ALL_TABS]
    results = await asyncio.gather(*tasks)
    RAM_CACHE = {tab: records for tab, records in results}
    return {"status": "success"}

@app.get("/")
def health_check():
    return {"status": "healthy", "provider": "Google GenAI SDK", "gemini_model": GEMINI_MODEL}

@app.get("/reload")
async def reload_data():
    return await load_sheet_data_async()

class ChatRequest(BaseModel):
    messages: list
    role: str = "Khach_Hang"

def extract_user_text(event: dict) -> str:
    if isinstance(event.get("message"), dict):
        msg = event["message"]
        if msg.get("text"): return msg["text"]
        if msg.get("argumentText"): return msg["argumentText"]

    if isinstance(event.get("chat"), dict):
        chat = event["chat"]
        if isinstance(chat.get("messagePayload"), dict) and isinstance(chat["messagePayload"].get("message"), dict):
            m = chat["messagePayload"]["message"]
            if m.get("text"): return m["text"]
            if m.get("argumentText"): return m["argumentText"]
        if isinstance(chat.get("message"), dict):
            m = chat["message"]
            if m.get("text"): return m["text"]
            if m.get("argumentText"): return m["argumentText"]

    def deep_search(obj):
        if isinstance(obj, dict):
            if "argumentText" in obj and isinstance(obj["argumentText"], str) and obj["argumentText"].strip():
                return obj["argumentText"]
            if "text" in obj and isinstance(obj["text"], str) and obj["text"].strip():
                if not obj["text"].startswith("spaces/"):
                    return obj["text"]
            for k, v in obj.items():
                res = deep_search(v)
                if res: return res
        return ""

    return deep_search(event)

def get_high_precision_knowledge(query: str, role: str) -> str:
    accessible_tabs = ALL_TABS if role == "Sale" else TABS_PUBLIC
    query_lower = query.lower()
    stop_words = {"mình", "có", "bị", "được", "không", "cho", "với", "là", "và", "nhé", "ạ", "cần", "giúp", "tôi", "xin", "lỗi", "máy", "thế", "nào", "bao", "nhiêu", "thông", "số", "in", "qua", "đã", "ok"}
    words = [w for w in query_lower.split() if len(w) > 1 and w not in stop_words]
    if not words: words = [query_lower]

    scored_rows = []
    for tab in accessible_tabs:
        for row in RAM_CACHE.get(tab, []):
            row_text = " ".join(str(v).lower() for v in row.values())
            score = 0
            dev_name = str(row.get("Ten_Thiet_Bi", row.get("Loai_Thiet_Bi", ""))).lower()
            for w in words:
                if len(w) >= 3 and w in dev_name: score += 50
                elif w in row_text: score += 3
            if score > 0: scored_rows.append((score, tab, row))

    scored_rows.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored_rows[:3]

    knowledge_text = ""
    for score, tab, row in top_matches:
        knowledge_text += f"\n=== DỮ LIỆU TỪ TAB [{tab}] ===\n"
        for key, value in row.items():
            if value: knowledge_text += f"- {key}: {value}\n"
    return knowledge_text

def build_smart_system_prompt(knowledge_context: str) -> str:
    return f"""
    Bạn là Trợ Lý KHO Sapo – Chuyên gia IT cao cấp hỗ trợ kỹ thuật thiết bị Sapo. Bạn phải thông minh, linh hoạt, biết tư duy liên kết dữ liệu.

    🎯 QUY TẮC PHẢN HỒI:
    1. Tư duy liên kết, hướng dẫn ngắn gọn, trực diện, thân thiện. Xưng "Em", gọi "Anh/chị". Dùng gạch đầu dòng, in đậm bước quan trọng.
    2. LUẬT THÉP CHỐNG BỊA ĐẶT: CẤM tuyệt đối bịa ra đường link website hoặc số điện thoại. Chỉ cung cấp Link nếu Link đó CÓ TRONG kho dữ liệu bên dưới.

    KHO DỮ LIỆU GỐC CỦA SAPO:
    {knowledge_context}
    """

def call_genai_sync(system_prompt: str, user_msg: str) -> str:
    if not genai_client:
        return "👋 Dạ em là Trợ Lý KHO Sapo. Anh/chị cần hỗ trợ tra cứu thiết bị nào ạ?"
    try:
        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_msg,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
                max_output_tokens=1000,
            )
        )
        return response.text
    except Exception as e:
        return f"Dạ hệ thống đang bận, anh/chị vui lòng thử lại nhé! ({str(e)[:40]})"

def wrap_gsuite_addon_response(text_message: str) -> dict:
    clean_text = re.sub(r'\[(.*?)\]\((https?://.*?)\)', r'\1 (\2)', text_message)
    return {
        "hostAppDataAction": {
            "chatDataAction": {
                "createMessageAction": {
                    "message": {
                        "text": clean_text
                    }
                }
            }
        }
    }

@app.post("/chat")
async def chat_stream(req: ChatRequest):
    latest_msg = req.messages[-1]["text"] if req.messages else ""
    clean_q = re.sub(r'[^\w\s]', '', latest_msg.lower()).strip()
    quick_greetings = ["chào", "chào bạn", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em"]
    if clean_q in quick_greetings:
        async def greeting_gen():
            yield "Xin chào! Em là **Trợ Lý KHO Sapo**. Anh/chị cần hỗ trợ tra cứu thông số thiết bị hay cài đặt máy in nào ạ?"
        return StreamingResponse(greeting_gen(), media_type="text/plain")

    focused_knowledge = get_high_precision_knowledge(latest_msg, req.role)
    system_instruction = build_smart_system_prompt(focused_knowledge)

    if not genai_client:
        return StreamingResponse(iter(["Dạ cấu hình GEMINI_API_KEY chưa được thiết lập."]), media_type="text/plain")

    async def generate_genai_stream():
        try:
            response_stream = genai_client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=latest_msg,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                    max_output_tokens=1000,
                )
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception:
            fallback_text = call_genai_sync(system_instruction, latest_msg)
            yield fallback_text

    return StreamingResponse(generate_genai_stream(), media_type="text/plain")

@app.post("/google-chat")
async def google_chat_webhook(request: Request):
    try:
        event = await request.json()
        user_message = extract_user_text(event)
        cleaned_message = re.sub(r'<.*?>', '', user_message).replace("@Trợ Lý KHO Sapo", "").strip()

        event_type = event.get("type") or event.get("chat", {}).get("type") or ""

        if event_type == "ADDED_TO_SPACE":
            msg = "👋 Xin chào! Em là Trợ Lý KHO Sapo. Hãy gõ tên thiết bị hoặc câu hỏi để em hỗ trợ ngay 24/7!"
            return JSONResponse(content=wrap_gsuite_addon_response(msg))

        quick_greetings = ["chào", "chào bạn", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em", "chao ban nhe"]
        if not cleaned_message or cleaned_message.lower() in quick_greetings:
            msg = "👋 Xin chào! Em là Trợ Lý KHO Sapo. Anh/chị cần hỗ trợ tra cứu thông số máy in hay cài đặt thiết bị nào ạ?"
            return JSONResponse(content=wrap_gsuite_addon_response(msg))

        focused_knowledge = get_high_precision_knowledge(cleaned_message, role="Sale")
        system_instruction = build_smart_system_prompt(focused_knowledge)

        ai_response = call_genai_sync(system_instruction, cleaned_message)

        return JSONResponse(content=wrap_gsuite_addon_response(ai_response))

    except Exception:
        msg = "Dạ em đã nhận thông tin. Anh/chị cần tra cứu cài đặt hay khắc phục lỗi thiết bị nào ạ?"
        return JSONResponse(content=wrap_gsuite_addon_response(msg))
