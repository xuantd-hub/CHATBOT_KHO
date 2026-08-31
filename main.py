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

app = FastAPI(title="Trợ Lý KHO Sapo Super Intelligent Engine", version="160.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# CẤU HÌNH BIẾN MÔI TRƯỜNG (ĐÃ CẬP NHẬT CHUẨN GEMINI 3.6 FLASH)
# ------------------------------------------------------------------------------
SHEET_ID = os.getenv("SHEET_ID", "1ZMq0mTiQTDiP92UPaOIv39Q17WJXDiuvrcyYwfs7_Ag").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()

ACTIVE_GROQ_MODEL = "llama-3.3-70b-versatile"
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

# ------------------------------------------------------------------------------
# KHỞI TẠO VÀ ĐỒNG BỘ DỮ LIỆU
# ------------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global HTTP_CLIENT
    HTTP_CLIENT = httpx.AsyncClient(
        timeout=httpx.Timeout(8.0, read=12.0),
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
    )
    asyncio.create_task(load_sheet_data_async())
    await discover_active_groq_model()

@app.on_event("shutdown")
async def shutdown_event():
    if HTTP_CLIENT:
        await HTTP_CLIENT.aclose()

async def discover_active_groq_model():
    global ACTIVE_GROQ_MODEL
    if not GROQ_API_KEY:
        return

    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    try:
        res = await HTTP_CLIENT.get(url, headers=headers, timeout=4.0)
        if res.status_code == 200:
            models_data = res.json().get("data", [])
            model_ids = [m["id"] for m in models_data]
            preferred_order = [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "llama3-8b-8192"
            ]
            for pref in preferred_order:
                if pref in model_ids:
                    ACTIVE_GROQ_MODEL = pref
                    return
            if model_ids:
                ACTIVE_GROQ_MODEL = model_ids[0]
    except Exception:
        pass

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
    return {
        "status": "healthy", 
        "version": "160.1",
        "active_groq_model": ACTIVE_GROQ_MODEL,
        "backup_gemini_model": GEMINI_MODEL,
        "has_gemini_backup": bool(GEMINI_API_KEY)
    }

@app.get("/reload")
async def reload_data():
    return await load_sheet_data_async()

class ChatRequest(BaseModel):
    messages: list
    role: str = "Khach_Hang"

# ------------------------------------------------------------------------------
# TRÍCH XUẤT NỘI DUNG VÀ TÌM KIẾM TRI THỨC KHO
# ------------------------------------------------------------------------------
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
Bạn là Trợ Lý KHO Sapo – Chuyên gia IT hỗ trợ kỹ thuật thiết bị Sapo cực kỳ THÔNG MINH, TINH TẾ và LỊCH SỰ.

🎯 QUY TẮC PHẢN HỒI THÔNG MINH:
1. **Trực diện, chuyên nghiệp**: Xưng "Em", gọi "Anh/chị". Trình bày cấu trúc rõ ràng với gạch đầu dòng, in đậm tên máy/bước quan trọng.
2. **NẾU CÂU HỎI CHỈ LÀ TÊN THIẾT BỊ NÓI CHUNG (VD: "spr02", "k200l"):**
   - Hãy tóm tắt ngắn gọn loại máy, sau đó trình bày các mục quan trọng (Lỗi thường gặp & Cách xử lý, Link Driver/Tài liệu cài đặt) một cách mạch lạc.
3. **LUẬT THÉP CHỐNG BỊA ĐẶT (BẮT BUỘC):**
   - KHÔNG TỰ BỊA RA ĐƯỜNG LINK (URL) VÀ SỐ ĐIỆN THOẠI.
   - Chỉ đính kèm Link Driver/Tài liệu nếu Link đó CÓ TRONG "KHO DỮ LIỆU GỐC" bên dưới.
   - Trình bày đường link dạng: `[Tên hiển thị](URL)`.

KHO DỮ LIỆU GỐC CỦA SAPO:
{knowledge_context}
"""

# ------------------------------------------------------------------------------
# HÀM GỌI GROQ HOẶC GEMINI 3.6 FLASH DỰ PHÒNG (NON-STREAMING)
# ------------------------------------------------------------------------------
async def call_llm_single(system_instruction: str, user_message: str) -> str:
    # 1. ƯU TIÊN GỌI GROQ
    if GROQ_API_KEY and ACTIVE_GROQ_MODEL:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": ACTIVE_GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.2,
            "max_tokens": 1000
        }
        try:
            res = await HTTP_CLIENT.post(url, headers=headers, json=payload, timeout=6.0)
            if res.status_code == 200:
                data = res.json()
                return data["choices"][0]["message"]["content"]
        except Exception:
            pass

    # 2. DỰ PHÒNG SANG GEMINI 3.6 FLASH NẾU GROQ LỖI HOẶC KHÔNG CÓ KEY
    if GEMINI_API_KEY:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }
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
        except Exception:
            pass

    return "👋 Dạ em là Trợ Lý KHO Sapo. Anh/chị cần hỗ trợ tra cứu thông số hay cài đặt thiết bị nào ạ?"

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

# ------------------------------------------------------------------------------
# 1. CỔNG WEB CHAT (/chat) - STREAMING GROQ -> GEMINI 3.6 FLASH FALLBACK
# ------------------------------------------------------------------------------
@app.post("/chat")
async def chat_stream(req: ChatRequest):
    latest_msg = req.messages[-1]["text"] if req.messages else ""
    clean_q = re.sub(r'[^\w\s]', '', latest_msg.lower()).strip()
    
    # ⚡ CÂU CHÀO OFFLINE TỰ ĐỘNG - TỐC ĐỘ 0 SECONDS
    quick_greetings = ["chào", "chào bạn", "chào bjan", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em", "chao ban nhe"]
    if clean_q in quick_greetings or "chào" in clean_q or "chao" in clean_q:
        async def greeting_gen():
            yield "Xin chào! Em là **Trợ Lý KHO Sapo**. Anh/chị cần hỗ trợ tra cứu thông số thiết bị hay cài đặt máy in nào ạ?"
        return StreamingResponse(greeting_gen(), media_type="text/plain")

    focused_knowledge = get_high_precision_knowledge(latest_msg, req.role)
    system_instruction = build_smart_system_prompt(focused_knowledge)

    async def generate_response_stream():
        # Thử Groq Stream
        if GROQ_API_KEY and ACTIVE_GROQ_MODEL:
            messages_payload = [{"role": "system", "content": system_instruction}]
            trimmed = req.messages[-5:] if len(req.messages) > 5 else req.messages
            for m in trimmed:
                role_type = "user" if m["role"] == "user" else "assistant"
                messages_payload.append({"role": role_type, "content": m["text"]})

            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": ACTIVE_GROQ_MODEL,
                "messages": messages_payload,
                "temperature": 0.2,
                "max_tokens": 1000,
                "stream": True
            }
            try:
                success_groq = False
                async with HTTP_CLIENT.stream("POST", url, headers=headers, json=payload, timeout=8.0) as response:
                    if response.status_code == 200:
                        async for line in response.aiter_lines():
                            if line and line.startswith("data: "):
                                data_str = line[6:].strip()
                                if data_str == "[DONE]": break
                                try:
                                    data_json = json.loads(data_str)
                                    choices = data_json.get("choices", [])
                                    if choices:
                                        chunk = choices[0].get("delta", {}).get("content", "")
                                        if chunk:
                                            success_groq = True
                                            yield chunk
                                except Exception: pass
                        if success_groq:
                            return
            except Exception: pass

        # Nếu Groq thất bại, tự động Fallback sang Gemini 3.6 Flash
        fallback_ans = await call_llm_single(system_instruction, latest_msg)
        yield fallback_ans

    return StreamingResponse(generate_response_stream(), media_type="text/plain")

# ------------------------------------------------------------------------------
# 2. CỔNG GOOGLE CHAT BOT (/google-chat)
# ------------------------------------------------------------------------------
@app.post("/google-chat")
async def google_chat_webhook(request: Request):
    try:
        event = await request.json()
        user_message = extract_user_text(event)
        cleaned_message = re.sub(r'<.*?>', '', user_message).replace("@Trợ Lý KHO Sapo", "").strip()

        event_type = event.get("type") or event.get("chat", {}).get("type") or ""
        if event_type == "ADDED_TO_SPACE":
            return JSONResponse(content=wrap_gsuite_addon_response("👋 Xin chào! Em là Trợ Lý KHO Sapo. Hãy gõ tên thiết bị hoặc câu hỏi để em hỗ trợ ngay 24/7!"))

        quick_greetings = ["chào", "chào bạn", "chào bjan", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em"]
        if not cleaned_message or cleaned_message.lower() in quick_greetings or "chào" in cleaned_message.lower():
            return JSONResponse(content=wrap_gsuite_addon_response("👋 Xin chào! Em là Trợ Lý KHO Sapo. Anh/chị cần hỗ trợ tra cứu thông số máy in hay cài đặt thiết bị nào ạ?"))

        focused_knowledge = get_high_precision_knowledge(cleaned_message, role="Sale")
        system_instruction = build_smart_system_prompt(focused_knowledge)

        ai_response = await call_llm_single(system_instruction, cleaned_message)
        return JSONResponse(content=wrap_gsuite_addon_response(ai_response))

    except Exception:
        return JSONResponse(content=wrap_gsuite_addon_response("👋 Dạ em đã nhận thông tin. Anh/chị cần tra cứu cài đặt hay khắc phục lỗi thiết bị nào ạ?"))
