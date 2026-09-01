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

app = FastAPI(title="Trợ Lý KHO Sapo Direct Sheet Engine", version="490.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# CẤU HÌNH BIẾN MÔI TRƯỜNG & KHỞI TẠO BỘ NHỚ RAM
# ------------------------------------------------------------------------------
SHEET_ID = os.getenv("SHEET_ID", "1ZMq0mTiQTDiP92UPaOIv39Q17WJXDiuvrcyYwfs7_Ag").strip()
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "").strip()
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b").strip()
AVAILABLE_CEREBRAS_MODELS = []

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()

RAM_CACHE = {}
DOCS_TEXT_CACHE = {}
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
# HÀM CÀO GOOGLE DOCS (CÓ THÊM LOG KIỂM TRA QUYỀN TRUY CẬP)
# ------------------------------------------------------------------------------
async def fetch_google_doc_content(url: str) -> str:
    if not url or "docs.google.com/document" not in url: return ""
    match = re.search(r'/document/d/([a-zA-Z0-9_-]+)', url)
    if not match: return ""
    doc_id = match.group(1)
    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    try:
        res = await HTTP_CLIENT.get(export_url, timeout=8.0, follow_redirects=True)
        if res.status_code == 200:
            txt = res.text.strip()
            if len(txt) > 20 and "<html" not in txt.lower():
                return txt
    except Exception: pass
    return ""

# ------------------------------------------------------------------------------
# KHỞI TẠO HTTP CLIENT & ĐỌC DỮ LIỆU GOOGLE SHEET
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
    if not CEREBRAS_API_KEY: return
    try:
        res = await HTTP_CLIENT.get("https://api.cerebras.ai/v1/models", headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}"}, timeout=6.0)
        if res.status_code == 200:
            model_ids = [m["id"] for m in res.json().get("data", [])]
            AVAILABLE_CEREBRAS_MODELS = model_ids
            if "gpt-oss-120b" in model_ids: CEREBRAS_MODEL = "gpt-oss-120b"
            elif "gemma-4-31b" in model_ids: CEREBRAS_MODEL = "gemma-4-31b"
            elif model_ids: CEREBRAS_MODEL = model_ids[0]
    except Exception:
        CEREBRAS_MODEL = "gpt-oss-120b"

async def fetch_single_tab_raw(tab: str):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={tab}"
    try:
        res = await HTTP_CLIENT.get(url, timeout=8.0)
        if res.status_code == 200 and "text/csv" in res.headers.get("Content-Type", ""):
            df = pd.read_csv(io.BytesIO(res.content)).fillna("")
            records = [{str(k): str(v).strip() for k, v in row.items() if str(v).strip()} for _, row in df.iterrows()]
            return tab, [r for r in records if r]
    except Exception: pass
    return tab, []

async def load_sheet_data_async():
    global RAM_CACHE, DOCS_TEXT_CACHE
    results = await asyncio.gather(*(fetch_single_tab_raw(tab) for tab in ALL_TABS))
    RAM_CACHE = {tab: records for tab, records in results}
    
    # Quét link Google Doc
    doc_urls = set()
    for records in RAM_CACHE.values():
        for row in records:
            for val in row.values():
                if "docs.google.com/document" in str(val):
                    for m in re.findall(r'https?://docs\.google\.com/document/d/[a-zA-Z0-9_-]+[^\s"]*', str(val)):
                        doc_urls.add(m)
    
    for url in doc_urls:
        content = await fetch_google_doc_content(url)
        if content:
            DOCS_TEXT_CACHE[url] = content

    return {"status": "success", "cached_docs": len(DOCS_TEXT_CACHE)}

@app.get("/")
def health_check():
    return {
        "status": "healthy", 
        "version": "490.0", 
        "active_cerebras_model": CEREBRAS_MODEL,
        "cached_docs": len(DOCS_TEXT_CACHE)
    }

@app.get("/reload")
async def reload_data():
    return await load_sheet_data_async()

class ChatRequest(BaseModel):
    messages: list
    role: str = "Khach_Hang"

# ------------------------------------------------------------------------------
# LÀM SẠCH VĂN BẢN VÀ BỘ LỌC DỮ LIỆU RAG CHÍNH XÁC CAO
# ------------------------------------------------------------------------------
def clean_thinking_process(text: str) -> str:
    if "Here's a thinking process:" in text:
        parts = text.split("Here's a thinking process:")
        text = parts[-1]
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
            for v in obj.values():
                res = deep_search(v)
                if res: return res
        return ""

    return deep_search(event)

# ------------------------------------------------------------------------------
# THUẬT TOÁN TÌM KIẾM DỮ LIỆU ĐỊNH HƯỚNG TÊN THIẾT BỊ (EXACT DEVICE RAG)
# ------------------------------------------------------------------------------
def get_high_precision_knowledge(query: str, role: str) -> str:
    accessible_tabs = ALL_TABS if role == "Sale" else TABS_PUBLIC
    query_lower = query.lower()

    # Nhận diện thiết bị trong câu hỏi
    device_keywords = ["spr02", "spr01", "k200l", "k200u", "a868", "hprt", "spl01", "xp350b", "g8"]
    detected_device = None
    for dev in device_keywords:
        if dev in query_lower:
            detected_device = dev
            break

    scored_rows = []
    for tab in accessible_tabs:
        for row in RAM_CACHE.get(tab, []):
            row_text = " ".join(str(v).lower() for v in row.values())
            dev_field = str(row.get("Ten_Thiet_Bi", "")).lower() + " " + str(row.get("Tu_Khoa_Nhan_Dien", "")).lower()
            
            score = 0
            if detected_device:
                if detected_device in dev_field:
                    score += 500  # Khớp đúng thiết bị
                else:
                    continue  # Bỏ qua dòng của máy khác hoàn toàn
            else:
                score += 1

            # Lọc theo thao tác
            if "driver" in query_lower or "máy tính" in query_lower or "windows" in query_lower:
                if "driver" in row_text or "windown" in row_text or "máy tính" in row_text:
                    score += 50
            if "điện thoại" in query_lower or "xtest" in query_lower or "lan" in query_lower:
                if "điện thoại" in row_text or "xtest" in row_text or "lan" in row_text:
                    score += 50

            if score > 0:
                scored_rows.append((score, tab, row))

    scored_rows.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored_rows[:2]

    knowledge_text = ""
    for score, tab, row in top_matches:
        knowledge_text += f"\n=== DỮ LIỆU TỪ SHEET [{tab}] ===\n"
        for key, value in row.items():
            if value: 
                knowledge_text += f"- {key}: {value}\n"
                # Nạp nội dung Doc nếu cào thành công
                for doc_url, doc_text in DOCS_TEXT_CACHE.items():
                    if doc_url in str(value):
                        knowledge_text += f"\n📖 [NỘI DUNG TẤT CẢ BƯỚC CÀI TRONG FILE GOOGLE DOC {doc_url}]:\n{doc_text}\n"

    return knowledge_text

# ------------------------------------------------------------------------------
# SYSTEM PROMPT KHẮC NGHIỆT - CẤM TỰ BỊA BƯỚC THỦ CÔNG
# ------------------------------------------------------------------------------
def build_smart_system_prompt(knowledge_context: str) -> str:
    return f"""
Bạn là **Trợ Lý KHO Sapo** – Kỹ thuật viên IT cao cấp phụ trách phần cứng Sapo.

🎯 QUY TẮC PHẢN HỒI BÁM SÁT DỮ LIỆU (LUẬT THÉP):
1. **TRẢ LỜI ĐÚNG VÀ ĐỦ THEO NỘI DUNG TRONG KHO DỮ LIỆU:**
   - Trình bày chính xác quy trình cài đặt/sửa lỗi từ Kho dữ liệu bên dưới.
   - **TUYỆT ĐỐI CẤM SỬ DỤNG TRI THỨC BÊN NGOÀI:** Không tự suy đoán các bước Windows thủ công như "Vào Control Panel -> Devices and Printers", không tự bịa bước "In self-test giữ nút Feed" nếu trong Kho dữ liệu không yêu cầu!

2. **XUẤT ĐẦY ĐỦ CÁC ĐƯỜNG LINK:**
   - Trích xuất toàn bộ Link Driver Windows, Link Driver Mac, Link Video YouTube có trong dữ liệu và đính kèm ở cuối bài.

3. **GIAO TIẾP TỰ NHIÊN & ĐỘNG BỘ LUỒNG HỘI THOẠI:**
   - Đọc kỹ lịch sử chat. Nếu người dùng đã cung cấp tên thiết bị (VD: SPR02, K200L...), dùng ngay tên máy đó, KHÔNG HỎI LẠI!
   - Xưng "Em", gọi "Anh/chị". Dùng gạch đầu dòng rõ ràng, KHÔNG dùng bảng Markdown.

---

KHO DỮ LIỆU GỐC SAPO:
{knowledge_context}
"""

# ------------------------------------------------------------------------------
# HÀM GỌI LLM ĐA LƯỢT
# ------------------------------------------------------------------------------
async def call_llm_with_history(system_instruction: str, messages_list: list) -> str:
    messages_payload = [{"role": "system", "content": system_instruction}]
    for m in messages_list[-6:]:
        role_type = "user" if m.get("role") in ["user", "Khach_Hang"] else "assistant"
        messages_payload.append({"role": role_type, "content": m.get("text", "")})

    if CEREBRAS_API_KEY and CEREBRAS_MODEL:
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": CEREBRAS_MODEL,
            "messages": messages_payload,
            "temperature": 0.0,
            "max_tokens": 2000
        }
        try:
            res = await HTTP_CLIENT.post(url, headers=headers, json=payload, timeout=4.0)
            if res.status_code == 200:
                data = res.json()
                return clean_thinking_process(data["choices"][0]["message"]["content"])
        except Exception: pass

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
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2000}
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
    clean_text = re.sub(r'\*{2,3}', '*', clean_text)
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
                "temperature": 0.0,
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
# 2. CỔNG GOOGLE CHAT BOT (/google-chat)
# ------------------------------------------------------------------------------
@app.post("/google-chat")
async def google_chat_webhook(request: Request):
    try:
        event = await request.json()
        user_message = extract_user_text(event)
        cleaned_message = re.sub(r'<.*?>', '', user_message).replace("@Trợ Lý KHO Sapo", "").strip()

        space_id = event.get("space", {}).get("name") or event.get("user", {}).get("name") or "default_space"

        event_type = event.get("type") or event.get("chat", {}).get("type") or ""
        if event_type == "ADDED_TO_SPACE":
            return JSONResponse(content=wrap_gsuite_addon_response("👋 Xin chào! Em là Trợ Lý KHO Sapo. Hãy gõ tên thiết bị hoặc câu hỏi để em hỗ trợ ngay 24/7!"))

        quick_greetings = ["chào", "chào bạn", "chào bjan", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em"]
        if not cleaned_message or any(g == cleaned_message.lower() for g in quick_greetings) or "chào" in cleaned_message.lower():
            return JSONResponse(content=wrap_gsuite_addon_response("👋 Xin chào! Em là Trợ Lý KHO Sapo. Anh/chị cần hỗ trợ tra cứu thông số máy in hay cài đặt thiết bị nào ạ?"))

        if space_id not in GOOGLE_CHAT_HISTORY:
            GOOGLE_CHAT_HISTORY[space_id] = []
        
        GOOGLE_CHAT_HISTORY[space_id].append({"role": "user", "text": cleaned_message})
        if len(GOOGLE_CHAT_HISTORY[space_id]) > 10:
            GOOGLE_CHAT_HISTORY[space_id] = GOOGLE_CHAT_HISTORY[space_id][-10:]

        combined_user_query = " ".join([m["text"] for m in GOOGLE_CHAT_HISTORY[space_id] if m["role"] == "user"])

        focused_knowledge = get_high_precision_knowledge(combined_user_query, role="Sale")
        system_instruction = build_smart_system_prompt(focused_knowledge)

        ai_response = await call_llm_with_history(system_instruction, GOOGLE_CHAT_HISTORY[space_id])

        GOOGLE_CHAT_HISTORY[space_id].append({"role": "assistant", "text": ai_response})

        return JSONResponse(content=wrap_gsuite_addon_response(ai_response))

    except Exception:
        return JSONResponse(content=wrap_gsuite_addon_response("👋 Dạ em chào anh/chị! Em là Trợ Lý KHO Sapo. Anh/chị cần em hỗ trợ cài đặt hay tra cứu lỗi thiết bị nào ạ?"))
