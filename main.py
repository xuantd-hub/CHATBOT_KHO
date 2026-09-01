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

app = FastAPI(title="Trợ Lý KHO Sapo True Debug Engine", version="400.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# CẤU HÌNH BIẾN MÔI TRƯỜNG
# ------------------------------------------------------------------------------
SHEET_ID = os.getenv("SHEET_ID", "1ZMq0mTiQTDiP92UPaOIv39Q17WJXDiuvrcyYwfs7_Ag").strip()
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "").strip()
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "llama-3.3-70b").strip()
AVAILABLE_CEREBRAS_MODELS = []

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()

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
# KHỞI TẠO HTTP CLIENT & DÒ TÌM MODEL CEREBRAS
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
        print("🚨 [STARTUP] CRITICAL: CEREBRAS_API_KEY bị trống!")
        return

    url = "https://api.cerebras.ai/v1/models"
    headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}"}
    try:
        res = await HTTP_CLIENT.get(url, headers=headers, timeout=6.0)
        print(f"🔍 [STARTUP] CEREBRAS CHECK MODEL HTTP STATUS: {res.status_code}")
        if res.status_code == 200:
            models_data = res.json().get("data", [])
            model_ids = [m["id"] for m in models_data]
            AVAILABLE_CEREBRAS_MODELS = model_ids
            print(f"📋 [STARTUP] DANH SÁCH MODEL CEREBRAS TRẢ VỀ: {model_ids}")
            
            env_model = os.getenv("CEREBRAS_MODEL", "llama-3.3-70b").strip()
            if env_model in model_ids:
                CEREBRAS_MODEL = env_model
            elif "llama-3.3-70b" in model_ids:
                CEREBRAS_MODEL = "llama-3.3-70b"
            elif model_ids:
                CEREBRAS_MODEL = model_ids[0]
            print(f"✅ [STARTUP] ĐÃ KHÓA MODEL CEREBRAS: {CEREBRAS_MODEL}")
        else:
            print(f"🚨 [STARTUP] LỖI CEREBRAS API MODELS ({res.status_code}): {res.text}")
    except Exception as e:
        print(f"🚨 [STARTUP] EXCEPTION CHECK CEREBRAS MODEL: {str(e)}")

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
    except Exception as e:
        print(f"🚨 [SHEET ERROR] TAB {tab}: {str(e)}")
    return tab, []

async def load_sheet_data_async():
    global RAM_CACHE
    tasks = [fetch_single_tab_raw(tab) for tab in ALL_TABS]
    results = await asyncio.gather(*tasks)
    RAM_CACHE = {tab: records for tab, records in results}
    print("✅ [SHEET] Dữ liệu Google Sheet đã nạp xong RAM Cache")
    return {"status": "success"}

@app.get("/")
def health_check():
    return {
        "status": "healthy", 
        "version": "400.0",
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
# TRÍCH XUẤT VÀ LÀM SẠCH DỮ LIỆU
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

def clean_thinking_process(text: str) -> str:
    if "Here's a thinking process:" in text:
        parts = text.split("Here's a thinking process:")
        last_part = parts[-1]
        match = re.search(r'(Dạ\s+|Xin chào|Trợ Lý KHO|\*\*|1\.|- )', last_part)
        if match:
            return last_part[match.start():].strip()
    
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()

# ------------------------------------------------------------------------------
# PROMPT CHUẨN KHO SAPO
# ------------------------------------------------------------------------------
def build_smart_system_prompt(knowledge_context: str) -> str:
    return f"""
Bạn là **Trợ Lý KHO Sapo** – Chuyên gia IT cao cấp phụ trách kỹ thuật phần cứng Sapo cực kỳ THÔNG MINH, TINH TẾ và CHÍNH XÁC CÔNG NGHỆ.

🚨 LUẬT KHÓA PHÂN LOẠI THIẾT BỊ (CHỐNG LẪN LỘN CỰC KỲ QUAN TRỌNG):
1. **NHẬN DIỆN ĐÚNG LOẠI THIẾT BỊ ĐANG HỎI:**
   - **MÁY IN TEM NHÃN / MÃ VẠCH (Ví dụ: G8, SPL01, XP-350B, XP-420B, SP460BT...):**
     ➔ Đây là dòng máy chuyên dùng để IN TEM/MÃ VẠCH.
     ➔ **TUYỆT ĐỐI CẤM** gọi dòng máy này là "máy in hóa đơn" hoặc phát ngôn "Máy in hóa đơn này không in được giấy tem"!
   - **MÁY IN HÓA ĐƠN (Ví dụ: SPR01, SPR02, K200L, K200U...):**
     ➔ Đây là dòng máy chuyên dùng để IN HÓA ĐƠN KHỔ K80/K57.

2. **CẮT BỎ LỊCH SỬ MÁY CŨ (CONTEXT ISOLATION):**
   - Khi người dùng đổi sang hỏi tên máy mới (Ví dụ: vừa hỏi SPR02 xong chuyển sang hỏi G8), bạn PHẢI TẬP TRUNG 100% VÀO MÁY MỚI (G8).
   - KHÔNG ĐƯỢC mang cảnh báo hay đặc tính của máy cũ (SPR02) gán cho máy mới (G8).

3. **BẮT BỘ KIỂM TRA PHẦN CỨNG (HARDWARE GATEKEEPER):**
   - Nếu thiết bị trong kho dữ liệu ghi chỉ có cổng **USB / Máy tính** (Ví dụ: G8, K200U...):
     ➔ **TUYỆT ĐỐI KHÔNG DÙNG MENU "Cài đặt qua điện thoại"**.
     ➔ Nếu người dùng hỏi hoặc chọn "cài qua điện thoại/máy POS", từ chối ngay: *"Dạ thiết bị **[Tên máy]** là dòng máy in kết nối qua cổng USB với Máy tính, KHÔNG hỗ trợ kết nối in qua Điện thoại hay App mobile ạ."*

---

🎯 QUY TẮC PHẢN HỒI THÔNG MINH THEO KỊCH BẢN:

1. **CÂU HỎI CHỈ CÓ TÊN MÁY:**
   - Nếu máy CÓ hỗ trợ Điện thoại/LAN: Đưa ra 3 lựa chọn (1. Máy tính, 2. Điện thoại/Máy POS, 3. Sự cố).
   - Nếu máy CHỈ hỗ trợ USB (như G8): Chỉ đưa ra 2 lựa chọn (1. Máy tính, 2. Sự cố).

2. **CÂU HỎI RÕ Ý ĐỊNH:**
   - Trả lời thẳng vào giải pháp, trình bày ngắn gọn, gạch đầu dòng rõ ràng.
   - Đính kèm đầy đủ link tài liệu/driver/video từ KHO DỮ LIỆU.

3. **LUẬT THÉP CHỐNG BỊA ĐẶT:**
   - TRẢ LỜI 100% BẰNG TIẾNG VIỆT.
   - Chỉ được phép cung cấp Link Driver / Tài liệu nếu Link đó CÓ TRONG mục KHO DỮ LIỆU bên dưới.
   - Dùng xưng hô "Em" - gọi "Anh/chị". Tuyệt đối không vẽ bảng.

---

KHO DỮ LIỆU GỐC CỦA SAPO (Chỉ lấy thông số & Link từ đây):
{knowledge_context}
"""

# ------------------------------------------------------------------------------
# HÀM GỌI GEMINI 3.6 FLASH DỰ PHÒNG CÓ LOG NGUYÊN VĂN
# ------------------------------------------------------------------------------
async def call_gemini_with_retry(system_instruction: str, user_message: str) -> str:
    if not GEMINI_API_KEY:
        print("🚨 [GEMINI CALL] CRITICAL ERROR: Biến GEMINI_API_KEY bị trống!")
        return "👋 Dạ em chào anh/chị! Em là **Trợ Lý KHO Sapo**. Anh/chị cần em hỗ trợ cài đặt hay tra cứu thông tin cho thiết bị nào ạ?"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2000}
    }

    try:
        res = await HTTP_CLIENT.post(url, headers=headers, json=payload, timeout=8.0)
        print(f"📡 [GEMINI CALL] HTTP STATUS: {res.status_code}")
        if res.status_code == 200:
            data = res.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            cleaned = clean_thinking_process(raw_text)
            if cleaned: return cleaned
        else:
            print(f"🚨 [GEMINI CALL ERROR] Status {res.status_code} | Payload: {res.text}")
    except Exception as e:
        print(f"🚨 [GEMINI CALL EXCEPTION]: {str(e)}")

    return "👋 Dạ em chào anh/chị! Em là **Trợ Lý KHO Sapo**. Anh/chị cần em hỗ trợ cài đặt hay tra cứu thông tin cho thiết bị nào ạ?"

async def call_llm_single(system_instruction: str, user_message: str) -> str:
    if CEREBRAS_API_KEY and CEREBRAS_MODEL:
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": CEREBRAS_MODEL,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        try:
            res = await HTTP_CLIENT.post(url, headers=headers, json=payload, timeout=3.5)
            print(f"📡 [CEREBRAS CALL] HTTP STATUS: {res.status_code}")
            if res.status_code == 200:
                data = res.json()
                return clean_thinking_process(data["choices"][0]["message"]["content"])
            else:
                print(f"🚨 [CEREBRAS CALL ERROR] Status {res.status_code} | Response: {res.text}")
        except Exception as e:
            print(f"🚨 [CEREBRAS CALL EXCEPTION]: {str(e)}")

    return await call_gemini_with_retry(system_instruction, user_message)

def wrap_gsuite_addon_response(text_message: str) -> dict:
    clean_text = re.sub(r'\[(.*?)\]\((https?://.*?)\)', r'\1 (\2)', text_message)
    return {"text": clean_text}

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
    combined_query = user_msgs[-1] if user_msgs else latest_msg

    focused_knowledge = get_high_precision_knowledge(combined_query, req.role)
    system_instruction = build_smart_system_prompt(focused_knowledge)

    async def generate_response_stream():
        has_yielded = False
        
        if CEREBRAS_API_KEY and CEREBRAS_MODEL:
            messages_payload = [{"role": "system", "content": system_instruction}]
            trimmed = req.messages[-3:] if len(req.messages) > 3 else req.messages
            for m in trimmed:
                role_type = "user" if m["role"] == "user" else "assistant"
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
                async with HTTP_CLIENT.stream("POST", url, headers=headers, json=payload, timeout=cerebras_timeout) as response:
                    print(f"📡 [CEREBRAS STREAM] HTTP STATUS: {response.status_code}")
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
                                            has_yielded = True
                                            yield chunk
                                except Exception: pass
                        if has_yielded:
                            return
                    else:
                        print(f"🚨 [CEREBRAS STREAM ERROR] Status Code: {response.status_code}")
            except Exception as e:
                print(f"🚨 [CEREBRAS STREAM EXCEPTION]: {str(e)}")

        if not has_yielded:
            print("🔄 [FALLBACK] Cerebras không nhả token -> Kích hoạt Gemini Backup...")
            fallback_ans = await call_gemini_with_retry(system_instruction, combined_query)
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
        if not cleaned_message or any(g == cleaned_message.lower() for g in quick_greetings) or "chào" in cleaned_message.lower():
            return JSONResponse(content=wrap_gsuite_addon_response("👋 Xin chào! Em là Trợ Lý KHO Sapo. Anh/chị cần hỗ trợ tra cứu thông số máy in hay cài đặt thiết bị nào ạ?"))

        focused_knowledge = get_high_precision_knowledge(cleaned_message, role="Sale")
        system_instruction = build_smart_system_prompt(focused_knowledge)

        try:
            ai_response = await asyncio.wait_for(call_llm_single(system_instruction, cleaned_message), timeout=2.8)
        except asyncio.TimeoutError:
            print("🚨 [GOOGLE CHAT TIMEOUT] Quá 2.8s -> Trả câu chào dự phòng để giữ kết nối")
            ai_response = "👋 Dạ em chào anh/chị! Em đã nhận thông tin. Anh/chị cần tra cứu cài đặt hay khắc phục lỗi cho model thiết bị nào ạ?"

        return JSONResponse(content=wrap_gsuite_addon_response(ai_response))

    except Exception as e:
        print(f"🚨 [GOOGLE CHAT WEBHOOK ERROR]: {str(e)}")
        return JSONResponse(content=wrap_gsuite_addon_response("👋 Dạ em chào anh/chị! Em là Trợ Lý KHO Sapo. Anh/chị cần em hỗ trợ cài đặt hay tra cứu lỗi thiết bị nào ạ?"))
