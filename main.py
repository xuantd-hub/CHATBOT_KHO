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

app = FastAPI(title="Trợ Lý KHO Sapo Cerebras Perfect Engine", version="800.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# CẤU HÌNH CEREBRAS API & LƯU BỘ NHỚ RAM
# ------------------------------------------------------------------------------
SHEET_ID = os.getenv("SHEET_ID", "1ZMq0mTiQTDiP92UPaOIv39Q17WJXDiuvrcyYwfs7_Ag").strip()
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "").strip()
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b").strip()
AVAILABLE_CEREBRAS_MODELS = []

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()

RAM_CACHE = {}
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
# KHỞI TẠO CLIENT & DÒ MODEL CEREBRAS
# ------------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global HTTP_CLIENT
    HTTP_CLIENT = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, read=20.0),
        limits=httpx.Limits(max_keepalive_connections=30, max_connections=100)
    )
    await load_sheet_data_async()
    await discover_active_cerebras_models()

@app.on_event("shutdown")
async def shutdown_event():
    if HTTP_CLIENT:
        await HTTP_CLIENT.aclose()

async def discover_active_cerebras_models():
    global CEREBRAS_MODEL, AVAILABLE_CEREBRAS_MODELS
    if not CEREBRAS_API_KEY: return
    try:
        res = await HTTP_CLIENT.get("https://api.cerebras.ai/v1/models", headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}"}, timeout=5.0)
        if res.status_code == 200:
            model_ids = [m["id"] for m in res.json().get("data", [])]
            AVAILABLE_CEREBRAS_MODELS = model_ids
            for pref in ["gpt-oss-120b", "llama-3.3-70b", "llama3.1-70b", "gemma-4-31b"]:
                if pref in model_ids:
                    CEREBRAS_MODEL = pref
                    return
            if model_ids: CEREBRAS_MODEL = model_ids[0]
    except Exception:
        CEREBRAS_MODEL = "gpt-oss-120b"

async def fetch_single_tab_raw(tab: str):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={tab}"
    try:
        res = await HTTP_CLIENT.get(url, timeout=6.0)
        if res.status_code == 200 and "text/csv" in res.headers.get("Content-Type", ""):
            df = pd.read_csv(io.BytesIO(res.content)).fillna("")
            records = [{str(k): str(v).strip() for k, v in row.items() if str(v).strip()} for _, row in df.iterrows()]
            return tab, [r for r in records if r]
    except Exception: pass
    return tab, []

async def load_sheet_data_async():
    global RAM_CACHE
    results = await asyncio.gather(*(fetch_single_tab_raw(tab) for tab in ALL_TABS))
    RAM_CACHE = {tab: records for tab, records in results}
    return {"status": "success"}

@app.get("/")
def health_check():
    return {
        "status": "healthy", 
        "version": "800.0", 
        "engine": "Cerebras Dedicated",
        "active_cerebras_model": CEREBRAS_MODEL,
        "available_cerebras_models": AVAILABLE_CEREBRAS_MODELS,
        "has_cerebras_key": bool(CEREBRAS_API_KEY)
    }

@app.get("/reload")
async def reload_data():
    return await load_sheet_data_async()

class ChatRequest(BaseModel):
    messages: list
    role: str = "Khach_Hang"

# ------------------------------------------------------------------------------
# LÀM SẠCH CHỮ VÀ TRÍCH XUẤT CÂU HỎI
# ------------------------------------------------------------------------------
def clean_thinking_process(text: str) -> str:
    if "Here's a thinking process:" in text:
        text = text.split("Here's a thinking process:")[-1]
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
# TRÍCH XUẤT DỮ LIỆU TỰ NHIÊN TỪ SHEET (RAG NHẸ & CHÍNH XÁC)
# ------------------------------------------------------------------------------
def get_high_precision_knowledge(query: str, role: str) -> str:
    accessible_tabs = ALL_TABS if role == "Sale" else TABS_PUBLIC
    query_lower = query.lower()

    # Nhận diện tên thiết bị cụ thể
    device_models = ["spr02", "spr01", "k200l", "k200u", "a868", "hprt", "80fe", "spl01", "xp350b", "g8", "a160m"]
    detected_dev = None
    for dev in device_models:
        if dev in query_lower:
            detected_dev = dev
            break

    scored_rows = []
    for tab in accessible_tabs:
        for row in RAM_CACHE.get(tab, []):
            row_text = " ".join(str(v).lower() for v in row.values())
            dev_field = str(row.get("Ten_Thiet_Bi", "")).lower() + " " + str(row.get("Tu_Khoa_Nhan_Dien", "")).lower()
            
            score = 0
            if detected_dev:
                if detected_dev in dev_field: score += 500
                else: continue
            else:
                score += 1

            if score > 0:
                scored_rows.append((score, tab, row))

    scored_rows.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored_rows[:2]

    knowledge_text = ""
    for score, tab, row in top_matches:
        knowledge_text += f"\n=== DỮ LIỆU TỪ SHEET TAB [{tab}] ===\n"
        for key, value in row.items():
            if value: knowledge_text += f"- {key}: {value}\n"
    return knowledge_text

# ------------------------------------------------------------------------------
# SYSTEM PROMPT TỐI ƯU DÀNH RIÊNG CHO CEREBRAS (CHUẨN 100% THEO ẢNH MẪU)
# ------------------------------------------------------------------------------
def build_smart_system_prompt(knowledge_context: str) -> str:
    return f"""
Bạn là **Trợ Lý KHO Sapo** – Kỹ thuật viên hỗ trợ phần cứng Sapo cực kỳ THÔNG MINH, TINH TẾ và LỊCH SỰ.

🎯 QUY TẮC PHẢN HỒI BẮT BUỘC (TUÂN THỦ 100%):

📌 KỊCH BẢN 1: KHI NGƯỜI DÙNG NÓI CÂU CHUNG CHUNG / CHƯA NÓI TÊN MÁY HOẶC CHƯA CHỌN CÁCH CÀI
(Ví dụ: "mình cần cài máy in hóa đơn trên điện thoại", "cài máy in", "hướng dẫn cài máy in"):
👉 TUYỆT ĐỐI KHÔNG xả ra bài hướng dẫn dài dòng! Hãy trả lời CHÍNH XÁC theo khung mẫu dưới đây:

"Dạ, để hỗ trợ anh/chị cài đặt máy in hóa đơn trên điện thoại, em cần xác nhận thêm một chút thông tin để gửi hướng dẫn nhé:

1. **Anh/chị đang dùng máy in hãng nào ạ?** (Ví dụ: Xprinter (SPR02, K200L...), HPRT (80FE...), hay hãng khác?)

2. **Anh/chị muốn cài đặt theo cách nào?**
   - **Cách A:** Cài qua App XTEST (phổ biến nhất cho máy Xprinter).
   - **Cách B:** Cài trực tiếp trong Cài đặt của điện thoại (Android/iOS) mà không cần tải app ngoài.

👉 Ví dụ: \"Em dùng máy SPR02, muốn cài qua app XTEST\" hoặc \"Em dùng HPRT 80FE\".

Anh/chị cho em biết cụ thể để em gửi link hướng dẫn chi tiết ngay ạ! 🙏"

---

📌 KỊCH BẢN 2: KHI NGƯỜI DÙNG ĐÃ NÓI RÕ TÊN MÁY HOẶC ĐÃ CHỌN CÁCH CÀI
(Ví dụ: "cài trên app xtest nhé", "cài driver spr02 máy tính", "hướng dẫn cài spr02"):
👉 Trả lời ngắn gọn, đúng trọng tâm (chỉ 3-4 bước ngắn) và TRÍCH XUẤT ĐÚNG LINK TỪ KHO DỮ LIỆU BÊN DƯỚI:

"Dạ, em hỗ trợ anh/chị cài đặt in hóa đơn qua ứng dụng XTEST trên điện thoại ngay ạ.

Dưới đây là các bước và tài liệu hướng dẫn chi tiết:

📱 **Hướng dẫn cài đặt qua App XTEST:**
1. **Tải ứng dụng:** Anh/chị tải app XTEST từ App Store (iOS) hoặc CH Play (Android).
2. **Kết nối:** Mở app, chọn kết nối với máy in (thường là qua Wi-Fi hoặc Bluetooth tùy model).
3. **Cấu hình IP:** Trong app XTEST, anh/chị cần thiết lập địa chỉ IP cho máy in để kết nối với hệ thống Sapo.
4. **Kiểm tra:** In thử một trang test để đảm bảo máy in hoạt động bình thường.

📄 **Tài liệu hướng dẫn chi tiết (Văn bản):** <Gợi ý tên tài liệu> (<Chèn Link Google Doc lấy từ dữ liệu>)
🎥 **Video hướng dẫn trực quan:** <Gợi ý tên video> (<Chèn Link Video lấy từ dữ liệu>)

Anh/chị làm theo video hoặc tài liệu trên nhé. Nếu gặp khó khăn ở bước nào, anh/chị cứ nhắn em hỗ trợ thêm ạ!"

---

❌ LUẬT THÉP CẤM BỊA ĐẶT:
- TUYỆT ĐỐI CẤM tự bịa ra các bước Windows thủ công như "Control Panel -> Devices and Printers", "Add local printer", "Cutter Select", hay "Giữ nút FEED".
- CHỈ được đính kèm link thực tế CÓ TRONG KHO DỮ LIỆU bên dưới.
- Xưng "Em", gọi "Anh/chị".

KHO DỮ LIỆU GỐC CỦA SAPO:
{knowledge_context}
"""

# ------------------------------------------------------------------------------
# HÀM GỌI CEREBRAS LLM (CÓ GEMINI LÀM DỰ PHÒNG AN TOÀN)
# ------------------------------------------------------------------------------
async def call_llm_with_history(system_instruction: str, messages_list: list) -> str:
    messages_payload = [{"role": "system", "content": system_instruction}]
    for m in messages_list[-6:]:
        role_type = "user" if m.get("role") in ["user", "Khach_Hang"] else "assistant"
        messages_payload.append({"role": role_type, "content": m.get("text", "")})

    # 1. GỌI CEREBRAS API DEDICATED
    if CEREBRAS_API_KEY and CEREBRAS_MODEL:
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": CEREBRAS_MODEL,
            "messages": messages_payload,
            "temperature": 0.1,
            "max_tokens": 1200
        }
        try:
            res = await HTTP_CLIENT.post(url, headers=headers, json=payload, timeout=8.0)
            if res.status_code == 200:
                data = res.json()
                return clean_thinking_process(data["choices"][0]["message"]["content"])
        except Exception: pass

    # 2. DỰ PHÒNG GEMINI (NẾU CEREBRAS MẤT MẠNG TẠM THỜI)
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
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1200}
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
# 1. CỔNG WEB CHAT (/chat) - CEREBRAS STREAMING
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
        ans = await call_llm_with_history(system_instruction, req.messages)
        yield ans

    return StreamingResponse(generate_response_stream(), media_type="text/plain")

# ------------------------------------------------------------------------------
# 2. CỔNG GOOGLE CHAT BOT (/google-chat) - CEREBRAS ENGINE
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
        return JSONResponse(content=wrap_gsuite_addon_response("👋 Dạ em chào anh/chị! Em là Trợ Lý KHO Sapo. Anh/chị cần em hỗ trợ cài đặt hay tra cứu thiết bị nào ạ?"))
