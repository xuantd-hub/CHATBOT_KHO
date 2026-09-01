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

app = FastAPI(title="Trợ Lý KHO Sapo Full Unified Engine", version="1900.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# CẤU HÌNH BIẾN MÔI TRƯỜNG & RAM CACHE
# ------------------------------------------------------------------------------
SHEET_ID = os.getenv("SHEET_ID", "1ZMq0mTiQTDiP92UPaOIv39Q17WJXDiuvrcyYwfs7_Ag").strip()
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "").strip()
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gemma-4-31b").strip()
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
# KHỞI TẠO HTTP CLIENT
# ------------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global HTTP_CLIENT
    HTTP_CLIENT = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, read=25.0),
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
            gemma_prefs = ["gemma-4-31b", "gemma-4-31b-it", "google/gemma-4-31b-it", "gemma-31b-it", "llama-3.3-70b"]
            for pref in gemma_prefs:
                if pref in model_ids:
                    CEREBRAS_MODEL = pref
                    return
            if model_ids: CEREBRAS_MODEL = model_ids[0]
    except Exception:
        CEREBRAS_MODEL = "gemma-4-31b"

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
        "version": "1900.0", 
        "engine": "Full Unified Engine",
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
# LÀM SẠCH VĂN BẢN (KHỬ MÃ LATEX MŨI TÊN $\rightarrow$)
# ------------------------------------------------------------------------------
def clean_thinking_process(text: str) -> str:
    if "Here's a thinking process:" in text:
        text = text.split("Here's a thinking process:")[-1]
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'---+', '', text)
    text = re.sub(r'\$\\rightarrow\$|\\rightarrow|\$\\Rightarrow\$|\\Rightarrow', '➔', text)
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
# BÓC TÁCH THIẾT BỊ HOẶC LOẠI MÁY IN TỪ LỊCH SỬ CHAT
# ------------------------------------------------------------------------------
def extract_device_info_from_history(messages: list) -> tuple:
    device_models = ["spr02", "spr01", "k200l", "k200u", "a868", "hprt", "80fe", "spl01", "xp350b", "g8", "a160m", "xprinter", "imin"]
    receipt_keywords = ["hóa đơn", "bill", "tính tiền"]
    label_keywords = ["tem", "mã vạch", "barcode", "nhãn"]

    detected_model = ""
    detected_category = ""

    for m in reversed(messages):
        txt = m.get("text", "").lower()
        if not detected_model:
            for dev in device_models:
                if dev in txt:
                    detected_model = dev
                    break
        if not detected_category:
            for r_kw in receipt_keywords:
                if r_kw in txt:
                    detected_category = "máy in hóa đơn"
                    break
            for l_kw in label_keywords:
                if l_kw in txt:
                    detected_category = "máy in tem"
                    break

    return detected_model, detected_category

# ------------------------------------------------------------------------------
# TRÍCH XUẤT DỮ LIỆU RAG VỚI LỊCH SỬ GỘP TOÀN BỘ CÂU HỎI CỦA KHÁCH
# ------------------------------------------------------------------------------
def get_high_precision_knowledge(messages_list: list, role: str) -> tuple:
    accessible_tabs = ALL_TABS if role == "Sale" else TABS_PUBLIC
    
    # Gộp toàn bộ câu hỏi của khách trong phiên chat để duy trì từ khóa sự cố
    user_texts = [m.get("text", "") for m in messages_list if m.get("role") in ["user", "Khach_Hang"]]
    combined_user_text = " ".join(user_texts).lower()
    
    latest_msg = messages_list[-1]["text"] if messages_list else ""
    latest_lower = latest_msg.lower()

    detected_model, detected_category = extract_device_info_from_history(messages_list)
    has_device_info = bool(detected_model or detected_category)

    # Nhận diện Ý định dựa trên tin nhắn MỚI NHẤT và TOÀN BỘ phiên chat
    is_setup_intent = any(k in latest_lower for k in ["cài", "driver", "xtest", "app", "điện thoại", "máy tính", "kết nối", "lan", "ip", "setup"])
    is_error_intent = any(k in combined_user_text for k in ["lỗi", "không", "kẹt", "hư", "trắng", "mực", "cắt", "sự cố", "sửa", "ra mực", "không ra"])
    is_policy_intent = any(k in combined_user_text for k in ["bảo hành", "đổi trả", "chính sách", "thu hồi"])

    stop_words = {"mình", "có", "bị", "được", "không", "cho", "với", "là", "và", "nhé", "ạ", "cần", "giúp", "tôi", "xin", "lỗi", "thế", "nào", "bao", "nhiêu", "thông", "số", "qua", "đã", "ok", "rồi", "nhưng", "lại", "muốn"}
    words = [w for w in combined_user_text.split() if len(w) > 1 and w not in stop_words]

    scored_rows = []
    for tab in accessible_tabs:
        tab_bonus = 0
        if is_setup_intent and tab == "2_HUONG_DAN_CAI_DAT": tab_bonus = 500
        elif is_error_intent and tab == "1_THIET_BI_VA_LOI": tab_bonus = 500
        elif is_policy_intent and tab == "3_CHINH_SACH_SAPO": tab_bonus = 500

        for row in RAM_CACHE.get(tab, []):
            row_text = " ".join(str(v).lower() for v in row.values())
            score = tab_bonus
            
            if detected_model and detected_model in row_text:
                score += 300
            if detected_category and detected_category in row_text:
                score += 150
            
            for w in words:
                if w in row_text:
                    score += 25
            
            if score > 0:
                scored_rows.append((score, tab, row))

    scored_rows.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored_rows[:3]

    knowledge_text = f"HAS_DEVICE_INFO: {has_device_info}\nDETECTED_MODEL: {detected_model}\nDETECTED_CATEGORY: {detected_category}\n"
    for score, tab, row in top_matches:
        knowledge_text += f"\n=== DỮ LIỆU THUỘC TAB [{tab}] ===\n"
        for key, value in row.items():
            if value: 
                knowledge_text += f"- {key}: {value}\n"

    return knowledge_text, has_device_info

# ------------------------------------------------------------------------------
# SYSTEM PROMPT BẢO VỆ ĐỊNH DẠNG & BÁM SÁT 100% SHEET
# ------------------------------------------------------------------------------
def build_smart_system_prompt(knowledge_context: str, has_device_info: bool) -> str:
    return f"""
Bạn là **Trợ Lý KHO Sapo** – Trợ lý AI cao cấp, có tư duy logic sâu sắc và am hiểu kỹ thuật thiết bị Sapo.
Xưng hô: Xưng "Em", gọi "Anh/chị". Phong cách: Lịch sự, chuyên nghiệp, hành văn tự nhiên, rõ ràng.

🎯 BỘ QUY TẮC XỬ LÝ QUAN TRỌNG NHẤT (TUÂN THỦ 100%):

1. 🛑 QUY TẮC PHÁT HIỆN THIẾU LOẠI MÁY IN / MODEL MÁY (DEVICE CLARIFICATION RULE):
   - Nếu trong Kho dữ liệu báo `HAS_DEVICE_INFO: False` (nghĩa là người dùng CHƯA CUNG CẤP tên model máy và CHƯA NÓI RÕ là Máy in hóa đơn hay Máy in tem mã vạch):
   - **TUYỆT ĐỐI CẤM tự ý suy đoán người dùng đang dùng Máy in hóa đơn!**
   - **TUYỆT ĐỐI CẤM tự ý xả bài hướng dẫn xử lý lỗi hay bài cài đặt của Máy in hóa đơn!**
   - **BẮT BUỘC hỏi lại ngay 1 câu khoanh vùng phân loại máy:**
     "Dạ, để em đưa ra hướng dẫn khắc phục chính xác nhất, anh/chị cho em hỏi mình đang sử dụng loại máy in nào ạ?
      1. 🧾 **Máy in hóa đơn (In bill tính tiền):** Ví dụ SPR02, K200L, K200U, A160M...
      2. 🏷️ **Máy in tem mã vạch (In tem dán sản phẩm):** Ví dụ SPL01, XP-350B, G8...
      
      Anh/chị cho em xin tên model cụ thể ghi trên máy để em gửi hướng dẫn chuẩn 100% cho mình nhé!"

2. 🛑 TRƯỜNG HỢP 2: KHI ĐÃ CÓ TÊN MODEL MÁY HOẶC ĐÃ NÓI RÕ LOẠI MÁY IN (`HAS_DEVICE_INFO: True`):
   - **BÁM SÁT 100% NỘI DUNG SHEET:** BẮT BUỘC phải trích xuất chính xác từng câu, từng bước và link ảnh có trong cột `Cach_Khac_Phuc` của Dữ liệu bên dưới.
   - **TUYỆT ĐỐI CẤM TỰ BỊA KIẾN THỨC BÊN NGOÀI:** Không tự giải thích lý thuyết in nhiệt, đảo cuộn giấy hay nói về giấy tem nếu trong dữ liệu gốc của dòng đó không yêu cầu!

👉 🛑 QUY TẮC ĐÍNH KÈM LINK TỪNG BƯỚC (INLINE STEP-LINK RULE):
   - Nếu ở MỖI BƯỚC CÓ ĐÍNH KÈM LINK ẢNH / LINK VIDEO RIÊNG:
     -> AI BẮT BUỘC phải đặt link ảnh/video đó NGAY DƯỚI BƯỚC TƯƠNG ỨNG!
     -> TUYỆT ĐỐI KHÔNG GỘM NGUYÊN ĐỐNG LINK VỀ CUỐI BÀI!

🧠 QUY TẮC DUY TRÌ BỐI CẢNH (Context Memory):
   - Khi người dùng đã cung cấp tên model (VD: SPR02 hay SPL01) ở các câu trước -> BẮT BUỘC phải giữ nguyên bối cảnh model đó cho toàn bộ các câu hỏi phía sau.
   - Khi chuyển từ Sửa lỗi sang Cài đặt (hoặc ngược lại) -> Giữ nguyên model và dẫn dắt tự nhiên bằng 1 câu ngắn ở đầu bài trước khi xuất hướng dẫn.

🛑 LUẬT THÉP ĐỊNH DẠNG:
- TUYỆT ĐỐI CẤM sử dụng mã LaTeX toán học (như $\\rightarrow$, $\\Rightarrow$). Dùng dấu mũi tên "➔" hoặc "->".
- Không tự bịa bước Control Panel hay thao tác phần cứng nếu dữ liệu không có.
- Chỉ cung cấp đường link có thực trong Kho dữ liệu dưới đây.

---

KHO DỮ LIỆU GỐC SAPO:
{knowledge_context}
"""

# ------------------------------------------------------------------------------
# HÀM GỌI LLM CÓ BẢO VỆ CHỐNG TRẢ VỀ CÂU CHÀO KHI LỖI
# ------------------------------------------------------------------------------
async def call_llm_with_history(system_instruction: str, messages_list: list) -> str:
    messages_payload = [{"role": "system", "content": system_instruction}]
    for m in messages_list[-6:]:
        role_type = "user" if m.get("role") in ["user", "Khach_Hang"] else "assistant"
        messages_payload.append({"role": role_type, "content": m.get("text", "")})

    # 1. THỬ GỌI CEREBRAS API
    if CEREBRAS_API_KEY and CEREBRAS_MODEL:
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": CEREBRAS_MODEL,
            "messages": messages_payload,
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 2000
        }
        try:
            res = await HTTP_CLIENT.post(url, headers=headers, json=payload, timeout=10.0)
            if res.status_code == 200:
                data = res.json()
                return clean_thinking_process(data["choices"][0]["message"]["content"])
            else:
                print(f"⚠️ Cerebras lỗi status {res.status_code}: {res.text}")
        except Exception as e:
            print(f"⚠️ Cerebras Exception: {str(e)}")

    # 2. DỰ PHÒNG CHUYỂN SANG GEMINI NẾU CEREBRAS NGHỄN/BẬN
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
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2000}
        }
        try:
            res = await HTTP_CLIENT.post(url, headers=headers, json=payload, timeout=8.0)
            if res.status_code == 200:
                data = res.json()
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                return clean_thinking_process(raw_text)
            else:
                print(f"⚠️ Gemini lỗi status {res.status_code}: {res.text}")
        except Exception as e:
            print(f"⚠️ Gemini Exception: {str(e)}")

    return "⚠️ Hệ thống AI hiện đang bận hoặc quá tải lượt truy cập (Lỗi kết nối). Anh/chị vui lòng nhấn gửi lại câu hỏi sau vài giây giúp em nhé! 🙏"

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
    
    exact_quick_greetings = {"chào", "chào bạn", "chào bjan", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em", "chao ban nhe", "xin chào"}
    if clean_q in exact_quick_greetings:
        async def greeting_gen():
            yield "Xin chào! Em là **Trợ Lý KHO Sapo**. Anh/chị cần hỗ trợ tra cứu thông số thiết bị hay cài đặt máy in nào ạ?"
        return StreamingResponse(greeting_gen(), media_type="text/plain")

    focused_knowledge, has_device_info = get_high_precision_knowledge(req.messages, req.role)
    system_instruction = build_smart_system_prompt(focused_knowledge, has_device_info)

    async def generate_response_stream():
        ans = await call_llm_with_history(system_instruction, req.messages)
        yield ans

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

        exact_quick_greetings = {"chào", "chào bạn", "chào bjan", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em", "chao ban nhe", "xin chào"}
        clean_user_q = re.sub(r'[^\w\s]', '', cleaned_message.lower()).strip()
        if not cleaned_message or clean_user_q in exact_quick_greetings:
            return JSONResponse(content=wrap_gsuite_addon_response("👋 Xin chào! Em là Trợ Lý KHO Sapo. Anh/chị cần hỗ trợ tra cứu thông số máy in hay cài đặt thiết bị nào ạ?"))

        if space_id not in GOOGLE_CHAT_HISTORY:
            GOOGLE_CHAT_HISTORY[space_id] = []
        
        GOOGLE_CHAT_HISTORY[space_id].append({"role": "user", "text": cleaned_message})
        if len(GOOGLE_CHAT_HISTORY[space_id]) > 10:
            GOOGLE_CHAT_HISTORY[space_id] = GOOGLE_CHAT_HISTORY[space_id][-10:]

        focused_knowledge, has_device_info = get_high_precision_knowledge(GOOGLE_CHAT_HISTORY[space_id], role="Sale")
        system_instruction = build_smart_system_prompt(focused_knowledge, has_device_info)

        ai_response = await call_llm_with_history(system_instruction, GOOGLE_CHAT_HISTORY[space_id])

        GOOGLE_CHAT_HISTORY[space_id].append({"role": "assistant", "text": ai_response})

        return JSONResponse(content=wrap_gsuite_addon_response(ai_response))

    except Exception:
        return JSONResponse(content=wrap_gsuite_addon_response("⚠️ Hệ thống AI hiện đang bận xử lý. Anh/chị vui lòng nhấn gửi lại câu hỏi sau vài giây giúp em nhé! 🙏"))
