import os
import io
import json
import asyncio
import time
import re
import pandas as pd
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Trợ Lý KHO Sapo Stable Baseline Engine", version="4800.0")

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
GOOGLE_CHAT_LAST_ACTIVE = {} 

TABS_PUBLIC = [
    "1_THIET_BI_VA_LOI", 
    "2_HUONG_DAN_CAI_DAT", 
    "3_CHINH_SACH_SAPO", 
    "NHAN_DIEN_THIET_BI"
]
TAB_PRIVATE = "4_DU_LIEU_NOI_BO"
TAB_CONFIG = "0_CAI_DAT"
ALL_TABS = [TAB_CONFIG] + TABS_PUBLIC + [TAB_PRIVATE]

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

def get_auto_reset_minutes() -> int:
    cai_dat_records = RAM_CACHE.get(TAB_CONFIG, [])
    if not cai_dat_records:
        return 0
    for row in cai_dat_records:
        for k, v in row.items():
            if "reset" in str(k).lower() or "thoi_gian" in str(k).lower() or "gia_tri" in str(k).lower():
                val_str = str(v).strip()
                if val_str.isdigit():
                    return int(val_str)
    return 0

@app.get("/")
def health_check():
    return {
        "status": "healthy", 
        "version": "4800.0", 
        "engine": "Stable Baseline Engine",
        "auto_reset_minutes": get_auto_reset_minutes(),
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
# BÓC TÁCH THIẾT BỊ HOẶC HỆ ĐIỀU HÀNH TỪ LỊCH SỬ CHAT
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

def extract_platform_intent(messages: list) -> str:
    user_msgs = [m.get("text", "") for m in messages if m.get("role") in ["user", "Khach_Hang"]]
    if not user_msgs:
        return ""

    for msg in reversed(user_msgs[-3:]):
        txt = msg.lower()
        if any(k in txt for k in ["mac", "macbook", "macos"]):
            return "mac"
        if any(k in txt for k in ["windows", "win", "win10", "win11", "win7"]):
            return "windows"
        if any(k in txt for k in ["điện thoại", "mobile", "app", "xtest", "android", "ios", "iphone", "lan"]):
            return "mobile"
            
    return ""

# ------------------------------------------------------------------------------
# TRÍCH XUẤT DỮ LIỆU RAG NGUYÊN BẢN CHUẨN V2100
# ------------------------------------------------------------------------------
def get_high_precision_knowledge(messages_list: list, role: str) -> tuple:
    accessible_tabs = ALL_TABS if role == "Sale" else TABS_PUBLIC
    
    user_texts = [m.get("text", "") for m in messages_list if m.get("role") in ["user", "Khach_Hang"]]
    combined_user_text = " ".join(user_texts).lower()

    detected_model, detected_category = extract_device_info_from_history(messages_list)
    platform_intent = extract_platform_intent(messages_list)
    has_device_info = bool(detected_model or detected_category)

    is_setup_intent = any(k in combined_user_text for k in ["cài", "driver", "xtest", "app", "điện thoại", "máy tính", "kết nối", "lan", "ip", "setup", "windows", "mac"])
    is_error_intent = any(k in combined_user_text for k in ["lỗi", "không", "kẹt", "hư", "trắng", "mực", "cắt", "sự cố", "sửa", "ra mực", "không ra"])
    is_policy_intent = any(k in combined_user_text for k in ["bảo hành", "đổi trả", "chính sách", "thu hồi"])

    stop_words = {"mình", "có", "bị", "được", "không", "cho", "với", "là", "và", "nhé", "ạ", "cần", "giúp", "tôi", "xin", "lỗi", "thế", "nào", "bao", "nhiêu", "thông", "số", "qua", "đã", "ok", "rồi", "nhưng", "lại", "muốn"}
    words = [w for w in combined_user_text.split() if len(w) > 1 and w not in stop_words]

    scored_rows = []
    for tab in accessible_tabs:
        if tab == TAB_CONFIG: continue
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

            if tab == "2_HUONG_DAN_CAI_DAT" and platform_intent:
                row_thao_tac = str(row.get("Loai_Thao_Tac", "")).lower() + " " + str(row.get("Tu_Khoa_Nhan_Dien", "")).lower() + " " + str(row.get("Noi_Dung_Huong_Dan", "")).lower()
                if platform_intent == "mac" and "mac" in row_thao_tac:
                    score += 1000
                elif platform_intent == "windows" and ("win" in row_thao_tac or "windows" in row_thao_tac):
                    score += 1000
                elif platform_intent == "mobile" and any(k in row_thao_tac for k in ["app", "xtest", "lan", "điện thoại", "mobile"]):
                    score += 1000
            
            for w in words:
                if w in row_text:
                    score += 25
            
            if score > 0:
                scored_rows.append((score, tab, row))

    scored_rows.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored_rows[:3]

    knowledge_text = f"HAS_DEVICE_INFO: {has_device_info}\nDETECTED_MODEL: {detected_model}\nDETECTED_CATEGORY: {detected_category}\nPLATFORM_INTENT: {platform_intent}\n"
    for score, tab, row in top_matches:
        knowledge_text += f"\n=== DỮ LIỆU THUỘC TAB [{tab}] ===\n"
        for key, value in row.items():
            if value: 
                knowledge_text += f"- {key}: {value}\n"

    return knowledge_text, has_device_info

# ------------------------------------------------------------------------------
# SYSTEM PROMPT BẢO VỆ NGUYÊN VĂN LINK TỪ SHEET
# ------------------------------------------------------------------------------
def build_smart_system_prompt(knowledge_context: str, has_device_info: bool) -> str:
    return f"""
Bạn là **Trợ Lý KHO Sapo** – Trợ lý AI cao cấp, có tư duy logic sâu sắc và am hiểu kỹ thuật thiết bị Sapo.
Xưng hô: Xưng "Em", gọi "Anh/chị". Phong cách: Lịch sự, chuyên nghiệp, hành văn tự nhiên, rõ ràng.

🎨 QUY TẮC BỐ CỤC TRÌNH BÀY ĐẸP MẮT & DỄ ĐỌC:
- **Sử dụng Icon/Emoji sinh động** ở đầu các tiêu đề và từng bước thao tác (VD: 💻, 🍏, 📱, 🛠️, 📌, ✅, 👉, 📄, 🎥, ⚠️).
- **Chia nhỏ đoạn văn thoáng đãng**, dùng danh sách gạch đầu dòng (Bullet points).
- **In đậm các từ khóa quan trọng**, tên nút bấm, tên bước (VD: **Bước 1: Tải Driver**, **Link Driver Mac:**).

🛑 NGUYÊN TẮC XUẤT NỘI DUNG VÀ ĐƯỜNG LINK (BẮT BUỘC 100%):
1. **TRÍCH XUẤT ĐẦY ĐỦ VÀ CHÍNH XÁC NGUYÊN VĂN** nội dung các bước hướng dẫn và TẤT CẢ các đường link URL (bắt đầu bằng `https://...`) có trong Kho dữ liệu dưới đây.
2. NẾU trong Kho dữ liệu có các trường chứa link (như `Noi_Dung_Huong_Dan`, `Link_Driver_Win`, `Link_Driver_Mac`, `Link_Video_HD`, `Link_Anh_Truc_Tiep`), BẮT BUỘC PHẢI DÁN NGUYÊN VĂN TOÀN BỘ LINK ĐÓ RA BÀI VIẾT.
3. TUYỆT ĐỐI CẤM viết câu hứa "tải theo link dưới đây" rồi lại KHÔNG dán link!
4. TUYỆT ĐỐI CẤM tự nghĩ ra link giả (như example.com, sapo.vn/huong-dan...). Nếu trong Kho dữ liệu không có link nào, không tự viết dòng hứa dán link.
5. TUYỆT ĐỐI CẤM tự viết cụm từ "(Ví dụ)" hoặc "(ví dụ)" vào sau các đường link.

🎯 BỘ QUY TẮC XỬ LÝ THEO TRƯỜNG HỢP:

1. 🛑 KHI CHƯA CÓ LOẠI MÁY IN HOẶC MODEL MÁY (`HAS_DEVICE_INFO: False`):
   - **TUYỆT ĐỐI CẤM tự ý suy đoán loại máy in!**
   - **BẮT BUỘC hỏi lại ngay 1 câu khoanh vùng phân loại máy:**
     "Dạ, để em đưa ra hướng dẫn khắc phục chính xác nhất, anh/chị cho em hỏi mình đang sử dụng loại máy in nào ạ?
      1. 🧾 **Máy in hóa đơn (In bill tính tiền):** Ví dụ SPR02, K200L, K200U, A160M...
      2. 🏷️ **Máy in tem mã vạch (In tem dán sản phẩm):** Ví dụ SPL01, XP-350B, G8...
      
      Anh/chị cho em xin tên model cụ thể ghi trên máy để em gửi hướng dẫn chuẩn 100% cho mình nhé!"

2. 🛑 KHI ĐÃ CÓ MODEL MÁY HOẶC LOẠI MÁY (`HAS_DEVICE_INFO: True`):
   - **BÁM SÁT 100% NỘI DUNG SHEET:** BẮT BUỘC trích xuất chính xác từng câu, từng bước và TẤT CẢ các link thực tế có trong Dữ liệu bên dưới.
   - Khi Dữ liệu có bài hướng dẫn cài trên **Mac**, AI BẮT BUỘC xuất bài hướng dẫn Mac và link Driver Mac.
   - Khi Dữ liệu có bài hướng dẫn cài trên **Windows**, AI xuất bài Windows và link Driver Windows.

🧠 QUY TẮC DUY TRÌ BỐI CẢNH (Context Memory):
   - Khi người dùng đã cung cấp tên model (VD: SPR02 hay K200L) ở các câu trước -> BẮT BUỘC phải giữ nguyên bối cảnh model đó cho toàn bộ các câu hỏi phía sau.

🛑 LUẬT THÉP ĐỊNH DẠNG:
- TUYỆT ĐỐI CẤM sử dụng mã LaTeX toán học (như $\\rightarrow$, $\\Rightarrow$). Dùng dấu mũi tên "➔" hoặc "->".
- Không tự bịa bước Control Panel hay thao tác phần cứng nếu dữ liệu không có.

---

KHO DỮ LIỆU GỐC SAPO:
{knowledge_context}
"""

# ------------------------------------------------------------------------------
# HÀM GỌI LLM CÓ BẢO VỆ CHỐNG LỖI KẾT NỐI
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
            "temperature": 0.1,
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
            else:
                print(f"⚠️ Gemini lỗi status {res.status_code}: {res.text}")
        except Exception as e:
            print(f"⚠️ Gemini Exception: {str(e)}")

    return "⚠️ Hệ thống AI hiện đang bận hoặc quá tải lượt truy cập (Lỗi kết nối). Anh/chị vui lòng nhấn gửi lại câu hỏi sau vài giây giúp em nhé! 🙏"

# ------------------------------------------------------------------------------
# HÀM WRAPPER DÀNH CHO GOOGLE CHAT CÓ CHÂN TRANG MẸO NỔI BẬT SIÊU MỎNG
# ------------------------------------------------------------------------------
def wrap_gsuite_addon_response(text_message: str, show_reset_note: bool = True) -> dict:
    clean_text = clean_thinking_process(text_message)
    clean_text = re.sub(r'\[(.*?)\]\((https?://.*?)\)', r'\1 (\2)', clean_text)
    clean_text = re.sub(r'\*{2,3}', '*', clean_text)
    
    if show_reset_note and "Em đã xóa bộ nhớ" not in clean_text:
        footer_note = (
            "\n\n───────────────────────────────\n"
            "> 💡 _Mẹo: Để đảm bảo dữ liệu chính xác nhất bạn hãy nhắn \"xóa lịch sử\" hoặc \"bắt đầu cuộc trò chuyện mới\" khi cần hỏi thông tin khác nhé!_"
        )
        clean_text += footer_note

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
# 2. CỔNG GOOGLE CHAT BOT (/google-chat) - CÓ LỆNH XÓA LỊCH SỬ & CHÂN TRANG MẸO
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
            return JSONResponse(content=wrap_gsuite_addon_response("👋 Xin chào! Em là Trợ Lý KHO Sapo. Hãy gõ tên thiết bị hoặc câu hỏi để em hỗ trợ ngay 24/7!", show_reset_note=False))

        clean_user_q = re.sub(r'[^\w\s]', '', cleaned_message.lower()).strip()

        # A. BẮT TẤT CẢ CÁC LỆNH XÓA LỊCH SỬ / BẮT ĐẦU TRÒ CHUYỆN MỚI
        reset_keywords = {
            "xóa lịch sử", "xoa lich su", 
            "chủ đề mới", "chu de moi",
            "bắt đầu cuộc trò chuyện mới", "bat dau cuoc tro chuyen moi",
            "bắt đầu trò chuyện mới", "bat dau tro chuyen moi",
            "xóa lịch sử cuộc trò chuyện", "xoa lich su cuoc tro chuyen"
        }
        if clean_user_q in reset_keywords:
            GOOGLE_CHAT_HISTORY[space_id] = []
            GOOGLE_CHAT_LAST_ACTIVE[space_id] = time.time()
            return JSONResponse(content=wrap_gsuite_addon_response("🧹 Em đã xóa bộ nhớ bối cảnh cuộc trò chuyện! Anh/chị cần em hỗ trợ cài đặt hay xử lý lỗi thiết bị nào mới ạ? 😊", show_reset_note=False))

        exact_quick_greetings = {"chào", "chào bạn", "chào bjan", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em", "chao ban nhe", "xin chào"}
        if not cleaned_message or clean_user_q in exact_quick_greetings:
            return JSONResponse(content=wrap_gsuite_addon_response("👋 Xin chào! Em là Trợ Lý KHO Sapo. Anh/chị cần hỗ trợ tra cứu thông số máy in hay cài đặt thiết bị nào ạ?", show_reset_note=False))

        # B. TỰ ĐỘNG XÓA LỊCH SỬ NẾU ĐỂ LÂU KHÔNG TƯƠNG TÁC
        now = time.time()
        last_active = GOOGLE_CHAT_LAST_ACTIVE.get(space_id, 0)
        reset_minutes = get_auto_reset_minutes()

        if reset_minutes > 0 and last_active > 0:
            timeout_seconds = reset_minutes * 60
            if (now - last_active) > timeout_seconds:
                GOOGLE_CHAT_HISTORY[space_id] = []

        GOOGLE_CHAT_LAST_ACTIVE[space_id] = now

        # C. LƯU LỊCH SỬ VÀ TRÍCH XUẤT RAG CHUẨN ĐỒNG BỘ
        if space_id not in GOOGLE_CHAT_HISTORY:
            GOOGLE_CHAT_HISTORY[space_id] = []
        
        GOOGLE_CHAT_HISTORY[space_id].append({"role": "user", "text": cleaned_message})
        if len(GOOGLE_CHAT_HISTORY[space_id]) > 10:
            GOOGLE_CHAT_HISTORY[space_id] = GOOGLE_CHAT_HISTORY[space_id][-10:]

        focused_knowledge, has_device_info = get_high_precision_knowledge(GOOGLE_CHAT_HISTORY[space_id], role="Sale")
        system_instruction = build_smart_system_prompt(focused_knowledge, has_device_info)

        ai_response = await call_llm_with_history(system_instruction, GOOGLE_CHAT_HISTORY[space_id])

        GOOGLE_CHAT_HISTORY[space_id].append({"role": "assistant", "text": ai_response})

        return JSONResponse(content=wrap_gsuite_addon_response(ai_response, show_reset_note=True))

    except Exception:
        return JSONResponse(content=wrap_gsuite_addon_response("⚠️ Hệ thống AI hiện đang bận xử lý. Anh/chị vui lòng nhắn gửi lại câu hỏi sau vài giây giúp em nhé! 🙏", show_reset_note=False))
