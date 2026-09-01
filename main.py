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

app = FastAPI(title="Trợ Lý KHO Sapo Unified Intelligent Engine", version="450.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# CẤU HÌNH BIẾN MÔI TRƯỜNG & BỘ NHỚ HỘI THOẠI LƯU BỞI SPACE ID
# ------------------------------------------------------------------------------
SHEET_ID = os.getenv("SHEET_ID", "1ZMq0mTiQTDiP92UPaOIv39Q17WJXDiuvrcyYwfs7_Ag").strip()
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "").strip()
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b").strip()
AVAILABLE_CEREBRAS_MODELS = []

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()

RAM_CACHE = {}
# Bộ nhớ lưu lịch sử 10 câu gần nhất cho mỗi phòng chat trên Google Chat
GOOGLE_CHAT_HISTORY = {} 

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
# KHỞI TẠO HTTP CLIENT & DÒ MODEL CEREBRAS
# ------------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global HTTP_CLIENT
    HTTP_CLIENT = httpx.AsyncClient(
        timeout=httpx.Timeout(15.0, read=45.0),
        limits=httpx.Limits(max_keepalive_connections=30, max_connections=100)
    )
    asyncio.create_task(load_sheet_data_async())
    await discover_active_cerebras_models()

@app.on_event("shutdown")
async def shutdown_event():
    if HTTP_CLIENT:
        await HTTP_CLIENT.aclose()

async def discover_active_cerebras_models():
    global CEREBRAS_MODEL, AVAILABLE_CEREBRAS_MODELS
    if not CEREBRAS_API_KEY:
        return

    url = "https://api.cerebras.ai/v1/models"
    headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}"}
    try:
        res = await HTTP_CLIENT.get(url, headers=headers, timeout=6.0)
        if res.status_code == 200:
            models_data = res.json().get("data", [])
            model_ids = [m["id"] for m in models_data]
            AVAILABLE_CEREBRAS_MODELS = model_ids
            if "gpt-oss-120b" in model_ids:
                CEREBRAS_MODEL = "gpt-oss-120b"
            elif "gemma-4-31b" in model_ids:
                CEREBRAS_MODEL = "gemma-4-31b"
            elif model_ids:
                CEREBRAS_MODEL = model_ids[0]
        else:
            CEREBRAS_MODEL = "gpt-oss-120b"
    except Exception:
        CEREBRAS_MODEL = "gpt-oss-120b"

async def fetch_single_tab_raw(tab: str):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={tab}"
    try:
        res = await HTTP_CLIENT.get(url, timeout=8.0)
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
        "version": "450.0",
        "active_cerebras_model": CEREBRAS_MODEL,
        "available_cerebras_models": AVAILABLE_CEREBRAS_MODELS,
        "has_cerebras_key": bool(CEREBRAS_API_KEY),
        "has_gemini_key": bool(GEMINI_API_KEY)
    }

@app.get("/reload")
async def reload_data():
    return await load_sheet_data_async()

class ChatRequest(BaseModel):
    messages: list
    role: str = "Khach_Hang"

# ------------------------------------------------------------------------------
# LÀM SẠCH VĂN BẢN VÀ CHỐNG TRẮNG MÀN HÌNH
# ------------------------------------------------------------------------------
def clean_thinking_process(text: str) -> str:
    if "Here's a thinking process:" in text:
        parts = text.split("Here's a thinking process:")
        last_part = parts[-1]
        match = re.search(r'(Dạ\s+|Xin chào|Trợ Lý KHO|\*\*|1\.|- )', last_part)
        if match:
            return last_part[match.start():].strip()
    
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'---+', '', text)
    return text.strip()

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

# ------------------------------------------------------------------------------
# TRÍCH XUẤT DỮ LIỆU ĐA CHIỀU CHÍNH XÁC CAO (RAG SEARCH)
# ------------------------------------------------------------------------------
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
                if len(w) >= 2 and w in dev_name: score += 50
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

# ------------------------------------------------------------------------------
# SYSTEM PROMPT TỐI ƯU - THÔNG MINH, CHI TIẾT & BÁM SÁT GOOGLE SHEET
# ------------------------------------------------------------------------------
def build_smart_system_prompt(knowledge_context: str) -> str:
    return f"""
Bạn là **Trợ Lý KHO Sapo** – Kỹ thuật viên IT cao cấp phụ trách phần cứng Sapo (máy in hóa đơn, máy in tem mã vạch, máy quét, POS...).

🎯 TƯ DUY NGHỆ & PHONG THÁI GIAO TIẾP:
- Nhạy bén, thực chiến, am hiểu IT. Xưng "Em", gọi "Anh/chị".
- **Hội thoại liên tục theo luồng:** Luôn đọc kỹ toàn bộ LỊCH SỬ CHAT. Nếu người dùng đã nhắc tên thiết bị (VD: SPR02, K200L, G8...) ở các câu nói trước, TUYỆT ĐỐI KHÔNG HỎI LẠI tên máy nữa!

🎯 QUY TRÌNH HƯỚNG DẪN CHI TIẾT (KHÔNG CẮT BỚT BƯỚC):
1. **Nếu câu hỏi từ khóa chung (Chỉ có tên máy như "spr02", "k200l"):**
   - Hỏi khoanh vùng lịch sự:
     "Dạ thiết bị **[Tên máy]**, anh/chị đang cần em hỗ trợ mục nào dưới đây ạ?
     1. 💻 **Cài đặt Driver trên Máy tính** (Windows / Mac)
     2. 📱 **Cài đặt in qua Điện thoại / Máy POS** (App XTEST / Kết nối LAN / Đổi IP)
     3. 🛠️ **Khắc phục sự cố** (Không cắt giấy, in ra giấy trắng, nghẽn mạng...)"

2. **Nếu câu hỏi rõ ý định (VD: "cài driver máy tính", "cài in qua điện thoại", "in bị mờ"):**
   - **Trả lời đầy đủ, chi tiết từng bước:** Trích xuất toàn bộ các bước kỹ thuật từ KHO DỮ LIỆU.
   - **Trích xuất ĐẦY ĐỦ LINK:** Cung cấp toàn bộ các đường link có trong dữ liệu (Link Driver, Link video YouTube, Link file hướng dẫn Word/PDF, Link app XTEST, Link đổi IP...).
   - **Nêu lưu ý quan trọng:** Đưa ra đầy đủ các cảnh báo phần cứng (VD: máy SPR02 không in được giấy tem, cấm lắp sai giấy...).

3. **Luật thép chống bịa đặt & Định dạng:**
   - 100% Tiếng Việt. KHÔNG xuất hiện câu suy nghĩ tiếng Anh.
   - CHỈ cung cấp đường link chính xác 100% có trong KHO DỮ LIỆU BÊN DƯỚI.
   - KHÔNG dùng bảng Markdown. Trình bày bằng emoji và gạch đầu dòng rõ ràng.

---

KHO DỮ LIỆU GỐC SAPO:
{knowledge_context}
"""

# ------------------------------------------------------------------------------
# HÀM GỌI LLM ĐA LƯỢT (ĐỒNG BỘ CẢ CEREBRAS & GEMINI)
# ------------------------------------------------------------------------------
async def call_llm_with_history(system_instruction: str, messages_list: list) -> str:
    messages_payload = [{"role": "system", "content": system_instruction}]
    for m in messages_list[-6:]:
        role_type = "user" if m.get("role") in ["user", "Khach_Hang"] else "assistant"
        messages_payload.append({"role": role_type, "content": m.get("text", "")})

    # 1. Thử Cerebras GPT-OSS-120B
    if CEREBRAS_API_KEY and CEREBRAS_MODEL:
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": CEREBRAS_MODEL,
            "messages": messages_payload,
            "temperature": 0.1,
            "max_tokens": 2000
        }
        try:
            res = await HTTP_CLIENT.post(url, headers=headers, json=payload, timeout=4.0)
            if res.status_code == 200:
                data = res.json()
                return clean_thinking_process(data["choices"][0]["message"]["content"])
        except Exception: pass

    # 2. Thử Gemini Dự Phòng
    if GEMINI_API_KEY:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        gemini_contents = []
        for m in messages_list[-6:]:
            role_type = "user" if m.get("role") in ["user", "Khach_Hang"] else "model"
            gemini_contents.append({"role": role_type, "parts": [{"text": m.get("text", "")}]})

        payload = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": gemini_contents,
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2000}
        }
        try:
            res = await HTTP_CLIENT.post(url, headers=headers, json=payload, timeout=8.0)
            if res.status_code == 200:
                data = res.json()
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                return clean_thinking_process(raw_text)
        except Exception: pass

    return "👋 Dạ em chào anh/chị! Em là **Trợ Lý KHO Sapo**. Anh/chị cần em hỗ trợ cài đặt hay tra cứu thiết bị nào ạ?"

def wrap_gsuite_addon_response(text_message: str) -> dict:
    clean_text = clean_thinking_process(text_message)
    clean_text = re.sub(r'\[(.*?)\]\((https?://.*?)\)', r'\1 (\2)', clean_text)
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
# 1. CỔNG WEB CHAT (/chat)
# ------------------------------------------------------------------------------
@app.post("/chat")
async def chat_stream(req: ChatRequest):
    latest_msg = req.messages[-1]["text"] if req.messages else ""
    clean_q = re.sub(r'[^\w\s]', '', latest_msg.lower()).strip()
    
    quick_greetings = ["chào", "chào bạn", "chào bjan", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em", "chao ban nhe"]
    if clean_q in quick_greetings or "chào" in clean_q or "chao" in clean_q:
        async def greeting_gen():
            yield "Xin chào! Em là **Trợ Lý KHO Sapo**. Anh/chị cần hỗ trợ tra cứu thông số thiết bị hay cài đặt máy in nào ạ?"
        return StreamingResponse(greeting_gen(), media_type="text/plain")

    # Ghép toàn bộ nội dung người dùng nói trong quá khứ để tìm kiếm dữ liệu chuẩn xác
    user_msgs = [m["text"] for m in req.messages if m.get("role") in ["user", "Khach_Hang"]]
    combined_query = " ".join(user_msgs)

    focused_knowledge = get_high_precision_knowledge(combined_query, req.role)
    system_instruction = build_smart_system_prompt(focused_knowledge)

    async def generate_response_stream():
        has_yielded = False
        
        if CEREBRAS_API_KEY and CEREBRAS_MODEL:
            messages_payload = [{"role": "system", "content": system_instruction}]
            trimmed = req.messages[-6:] if len(req.messages) > 6 else req.messages
            for m in trimmed:
                role_type = "user" if m["role"] in ["user", "Khach_Hang"] else "assistant"
                messages_payload.append({"role": role_type, "content": m["text"]})

            url = "https://api.cerebras.ai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": CEREBRAS_MODEL,
                "messages": messages_payload,
                "temperature": 0.1,
                "max_tokens": 2000,
                "stream": True
            }
            try:
                cerebras_timeout = httpx.Timeout(10.0, connect=2.5)
                full_stream_text = ""
                async with HTTP_CLIENT.stream("POST", url, headers=headers, json=payload, timeout=cerebras_timeout) as response:
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
                                        if chunk and not chunk.startswith("<think>"):
                                            full_stream_text += chunk
                                except Exception: pass
                        
                        clean_output = clean_thinking_process(full_stream_text)
                        if clean_output:
                            has_yielded = True
                            yield clean_output
                            return
            except Exception: pass

        if not has_yielded:
            fallback_ans = await call_llm_with_history(system_instruction, req.messages)
            yield fallback_ans

    return StreamingResponse(generate_response_stream(), media_type="text/plain")

# ------------------------------------------------------------------------------
# 2. CỔNG GOOGLE CHAT BOT (/google-chat) - ĐỘNG BỘ LỊCH SỬ ĐA LƯỢT MẠNH MẼ
# ------------------------------------------------------------------------------
@app.post("/google-chat")
async def google_chat_webhook(request: Request):
    try:
        event = await request.json()
        user_message = extract_user_text(event)
        cleaned_message = re.sub(r'<.*?>', '', user_message).replace("@Trợ Lý KHO Sapo", "").strip()

        # Định danh phòng chat duy nhất
        space_id = event.get("space", {}).get("name") or event.get("user", {}).get("name") or "default_space"

        event_type = event.get("type") or event.get("chat", {}).get("type") or ""
        if event_type == "ADDED_TO_SPACE":
            return JSONResponse(content=wrap_gsuite_addon_response("👋 Xin chào! Em là Trợ Lý KHO Sapo. Hãy gõ tên thiết bị hoặc câu hỏi để em hỗ trợ ngay 24/7!"))

        quick_greetings = ["chào", "chào bạn", "chào bjan", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em"]
        if not cleaned_message or any(g == cleaned_message.lower() for g in quick_greetings) or "chào" in cleaned_message.lower():
            return JSONResponse(content=wrap_gsuite_addon_response("👋 Xin chào! Em là Trợ Lý KHO Sapo. Anh/chị cần hỗ trợ tra cứu thông số máy in hay cài đặt thiết bị nào ạ?"))

        # Cập nhật tin nhắn của người dùng vào Session History
        if space_id not in GOOGLE_CHAT_HISTORY:
            GOOGLE_CHAT_HISTORY[space_id] = []
        
        GOOGLE_CHAT_HISTORY[space_id].append({"role": "user", "text": cleaned_message})
        if len(GOOGLE_CHAT_HISTORY[space_id]) > 10:
            GOOGLE_CHAT_HISTORY[space_id] = GOOGLE_CHAT_HISTORY[space_id][-10:]

        # Tổng hợp ngữ cảnh từ lịch sử để thực hiện RAG Search chính xác
        combined_user_query = " ".join([m["text"] for m in GOOGLE_CHAT_HISTORY[space_id] if m["role"] == "user"])

        focused_knowledge = get_high_precision_knowledge(combined_user_query, role="Sale")
        system_instruction = build_smart_system_prompt(focused_knowledge)

        # Gọi LLM với toàn bộ mảng lịch sử trò chuyện
        ai_response = await call_llm_with_history(system_instruction, GOOGLE_CHAT_HISTORY[space_id])

        # Lưu phản hồi của AI vào Session History
        GOOGLE_CHAT_HISTORY[space_id].append({"role": "assistant", "text": ai_response})

        return JSONResponse(content=wrap_gsuite_addon_response(ai_response))

    except Exception:
        return JSONResponse(content=wrap_gsuite_addon_response("👋 Dạ em chào anh/chị! Em là Trợ Lý KHO Sapo. Anh/chị cần em hỗ trợ cài đặt hay tra cứu lỗi thiết bị nào ạ?"))
