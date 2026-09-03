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

app = FastAPI(title="Trợ Lý KHO Sapo Strict Sheet Engine", version="4700.0")

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
        "version": "4700.0", 
        "engine": "Strict Sheet Engine",
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
# LÀM SẠCH VĂN BẢN VÀ LỌC XÓA TỰ ĐỘNG LINK GIẢ LẬP
# ------------------------------------------------------------------------------
def clean_thinking_process(text: str) -> str:
    if "Here's a thinking process:" in text:
        text = text.split("Here's a thinking process:")[-1]
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'---+', '', text)
    text = re.sub(r'\$\\rightarrow\$|\\rightarrow|\$\\Rightarrow\$|\\Rightarrow', '➔', text)
    return text.strip()

def restore_exact_urls(text: str, top_matches: list) -> str:
    text = re.sub(r'\s*\(\s*[Vv]í dụ:?\s*\)', '', text)

    url_regex = re.compile(r'https?://[^\s\)\>\]"\'\}]+')
    all_real_urls = []
    real_video_urls = []
    real_doc_urls = []

    if top_matches:
        for score, tab, row in top_matches:
            for k, v in row.items():
                val_str = str(v).strip()
                extracted = url_regex.findall(val_str)
                for u in extracted:
                    clean_u = u.rstrip('.,;')
                    if clean_u not in all_real_urls:
                        all_real_urls.append(clean_u)
                    if "youtube.com" in clean_u or "youtu.be" in clean_u:
                        if clean_u not in real_video_urls: real_video_urls.append(clean_u)
                    else:
                        if clean_u not in real_doc_urls: real_doc_urls.append(clean_u)

    found_urls = url_regex.findall(text)
    for found_url in found_urls:
        clean_found = found_url.rstrip('.,;')
        is_fake = (clean_found not in all_real_urls) or any(pkg in clean_found.lower() for pkg in ["example", "sapo.vn/huong-dan", "driver_win", "driver_mac", "1s_", "s-s-s"])

        if is_fake:
            if "youtube" in clean_found.lower() or "video" in clean_found.lower():
                if real_video_urls:
                    text = text.replace(clean_found, real_video_urls[0])
                else:
                    text = re.sub(r'^[^\n]*' + re.escape(clean_found) + r'[^\n]*\n?', '', text, flags=re.MULTILINE)
            else:
                if real_doc_urls:
                    text = text.replace(clean_found, real_doc_urls[0])
                elif all_real_urls:
                    text = text.replace(clean_found, all_real_urls[0])
                else:
                    text = re.sub(r'^[^\n]*' + re.escape(clean_found) + r'[^\n]*\n?', '', text, flags=re.MULTILINE)

    return text

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
# QUÉT BỐI CẢNH MODEL VÀ CATEGORY THÔNG MINH
# ------------------------------------------------------------------------------
def extract_device_info_from_history(messages: list) -> tuple:
    device_models = ["spr02", "spr01", "k200l", "k200u", "a868", "hprt", "80fe", "spl01", "xp350b", "g8", "a160m", "xprinter", "imin"]
    receipt_keywords = ["hóa đơn", "bill", "tính tiền"]
    label_keywords = ["tem", "mã vạch", "barcode", "nhãn"]

    user_msgs = [m.get("text", "") for m in messages if m.get("role") in ["user", "Khach_Hang"]]
    if not user_msgs:
        return "", ""

    latest_txt = user_msgs[-1].lower()

    latest_model = ""
    for dev in device_models:
        if dev in latest_txt:
            latest_model = dev
            break

    latest_category = ""
    for r_kw in receipt_keywords:
        if r_kw in latest_txt:
            latest_category = "máy in hóa đơn"
            break
    if not latest_category:
        for l_kw in label_keywords:
            if l_kw in latest_txt:
                latest_category = "máy in tem"
                break

    history_model = ""
    history_category = ""
    for m in reversed(user_msgs[:-1]):
        txt = m.lower()
        if not history_model:
            for dev in device_models:
                if dev in txt:
                    history_model = dev
                    break
        if not history_category:
            for r_kw in receipt_keywords:
                if r_kw in txt:
                    history_category = "máy in hóa đơn"
                    break
            for l_kw in label_keywords:
                if l_kw in txt:
                    history_category = "máy in tem"
                    break

    if latest_model:
        cat = latest_category or history_category or ""
        return latest_model, cat

    if latest_category and history_category and (latest_category != history_category):
        return "", latest_category

    if history_model:
        return history_model, latest_category or history_category

    return "", latest_category

def extract_platform_intent(messages: list) -> str:
    for m in reversed(messages):
        if m.get("role") in ["user", "Khach_Hang"]:
            txt = m.get("text", "").lower()
            if any(k in txt for k in ["mac", "macbook", "macos"]):
                return "mac"
            if any(k in txt for k in ["windows", "win", "win10", "win11", "win7"]):
                return "windows"
            if any(k in txt for k in ["điện thoại", "mobile", "app", "xtest", "android", "ios", "iphone", "lan"]):
                return "mobile"
    return ""

def extract_latest_action_intent_keywords(messages: list) -> str:
    for m in reversed(messages):
        if m.get("role") in ["user", "Khach_Hang"]:
            txt = m.get("text", "").lower()
            if any(k in txt for k in ["bảo hành", "đổi trả", "chính sách", "thu hồi", "tổng đài", "hotline", "sđt", "liên hệ", "địa chỉ", "kho", "sổ", "admin", "tỉnh"]):
                return "policy"
            if any(k in txt for k in ["lan", "ip", "mạng", "wifi", "app", "xtest", "điện thoại"]):
                return "lan_setup"
            if any(k in txt for k in ["driver", "cài", "setup", "máy tính", "windows", "mac"]):
                return "driver_setup"
            if any(k in txt for k in ["lỗi", "không", "kẹt", "hư", "trắng", "mực", "cắt", "sửa", "ra mực", "không ra"]):
                return "error_fix"
    return ""

async def extract_latest_action_intent(messages: list) -> str:
    user_msgs = [m.get("text", "") for m in messages if m.get("role") in ["user", "Khach_Hang"]]
    if not user_msgs:
        return ""
    
    combined_recent_text = " ".join(user_msgs[-3:]).strip()

    if HTTP_CLIENT and CEREBRAS_API_KEY and CEREBRAS_MODEL and combined_recent_text:
        try:
            router_prompt = f"""Bạn là bộ phân loại ý định hỗ trợ kỹ thuật cho Sapo.
Hãy phân loại đoạn hội thoại sau của khách hàng vào DUY NHẤT 1 trong các nhãn:
- policy: Bảo hành, đổi trả, số tổng đài, hotline, sđt liên hệ, địa chỉ kho, lịch làm việc, thông tin liên hệ Sapo, quy định chung.
- lan_setup: Cài đặt in qua mạng LAN, địa chỉ IP, kết nối Wifi, in qua App trên điện thoại/tablet (như xTest, Sapo App).
- driver_setup: Tải driver, cài máy tính Windows, cài Mac/macOS.
- error_fix: Báo lỗi thiết bị (không in được, kẹt giấy, in ra giấy trắng, không ra mực, kẹt dao cắt, hỏng hóc, sửa chữa).
- general: Chào hỏi hoặc câu hỏi chung.

Chỉ trả về DUY NHẤT tên nhãn (không viết thêm giải thích).
Đoạn hội thoại: "{combined_recent_text}"
Nhãn:"""

            res = await HTTP_CLIENT.post(
                "https://api.cerebras.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": CEREBRAS_MODEL,
                    "messages": [{"role": "user", "content": router_prompt}],
                    "temperature": 0.0,
                    "max_tokens": 10
                },
                timeout=1.2
            )
            if res.status_code == 200:
                intent_res = res.json()["choices"][0]["message"]["content"].strip().lower()
                for valid_intent in ["policy", "lan_setup", "driver_setup", "error_fix"]:
                    if valid_intent in intent_res:
                        return valid_intent
        except Exception:
            pass

    return extract_latest_action_intent_keywords(messages)

# ------------------------------------------------------------------------------
# MA TRẬN ĐIỀU HƯỚNG TRÍCH XUẤT DỮ LIỆU CHÍNH XÁC
# ------------------------------------------------------------------------------
async def get_high_precision_knowledge(messages_list: list, role: str) -> tuple:
    accessible_tabs = ALL_TABS if role == "Sale" else TABS_PUBLIC
    
    user_texts = [m.get("text", "") for m in messages_list if m.get("role") in ["user", "Khach_Hang"]]
    combined_user_text = " ".join(user_texts).lower()

    detected_model, detected_category = extract_device_info_from_history(messages_list)
    platform_intent = extract_platform_intent(messages_list)
    action_intent = await extract_latest_action_intent(messages_list)
    
    has_device_info = bool(detected_model) or (action_intent == "policy")

    stop_words = {"mình", "có", "bị", "được", "không", "cho", "với", "là", "và", "nhé", "ạ", "cần", "giúp", "tôi", "xin", "lỗi", "thế", "nào", "bao", "nhiêu", "thông", "số", "qua", "đã", "ok", "rồi", "nhưng", "lại", "muốn"}
    words = [w for w in combined_user_text.split() if len(w) > 1 and w not in stop_words]

    tech_rows = []    
    policy_rows = []  

    for tab in accessible_tabs:
        if tab == TAB_CONFIG: continue
        for row in RAM_CACHE.get(tab, []):
            row_text = " ".join(str(v).lower() for v in row.values())
            score = 0
            
            for w in words:
                if w in row_text:
                    score += 25

            if tab in ["3_CHINH_SACH_SAPO", "4_DU_LIEU_NOI_BO"]:
                score += 300  
                if action_intent == "policy":
                    score += 1000
                if score > 0:
                    policy_rows.append((score, tab, row))

            elif tab == "1_THIET_BI_VA_LOI":
                if detected_model and detected_model in row_text: score += 300
                if detected_category and detected_category in row_text: score += 150
                if action_intent == "error_fix": score += 1000
                if score > 0: tech_rows.append((score, tab, row))

            elif tab == "2_HUONG_DAN_CAI_DAT":
                if detected_model and detected_model in row_text: score += 300
                if detected_category and detected_category in row_text: score += 150
                row_thao_tac = str(row.get("Loai_Thao_Tac", "")).lower() + " " + str(row.get("Tu_Khoa_Nhan_Dien", "")).lower() + " " + str(row.get("Noi_Dung_Huong_Dan", "")).lower()
                if action_intent in ["lan_setup", "driver_setup"]: score += 1000
                if platform_intent == "mac" and "mac" in row_thao_tac: score += 300
                elif platform_intent == "windows" and ("win" in row_thao_tac or "windows" in row_thao_tac): score += 300
                if score > 0: tech_rows.append((score, tab, row))

            else:
                if detected_model and detected_model in row_text: score += 300
                if score > 0: tech_rows.append((score, tab, row))

    tech_rows.sort(key=lambda x: x[0], reverse=True)
    policy_rows.sort(key=lambda x: x[0], reverse=True)

    top_matches = []
    if action_intent == "policy":
        top_matches = policy_rows[:3] + tech_rows[:1]
    else:
        top_matches = tech_rows[:3] + policy_rows[:1]

    knowledge_text = f"HAS_DEVICE_INFO: {has_device_info}\nDETECTED_MODEL: {detected_model}\nDETECTED_CATEGORY: {detected_category}\nPLATFORM_INTENT: {platform_intent}\nACTION_INTENT: {action_intent}\n"
    for score, tab, row in top_matches:
        knowledge_text += f"\n=== DỮ LIỆU THUỘC TAB [{tab}] ===\n"
        for key, value in row.items():
            if value: 
                knowledge_text += f"- {key}: {value}\n"

    return knowledge_text, has_device_info, top_matches

# ------------------------------------------------------------------------------
# SYSTEM PROMPT (ÉP BÁM SÁT 100% NỘI DUNG SHEET & KHÔNG BỊA BƯỚC/LINK)
# ------------------------------------------------------------------------------
def build_smart_system_prompt(knowledge_context: str, has_device_info: bool) -> str:
    return f"""
Bạn là **Trợ Lý KHO Sapo** – Trợ lý AI cao cấp, có tư duy logic sâu sắc và am hiểu kỹ thuật thiết bị Sapo.
Xưng hô: Xưng "Em", gọi "Anh/chị". Phong cách: Lịch sự, chuyên nghiệp, hành văn tự nhiên, rõ ràng.

🛑 LUẬT NGUYÊN TẮC TỐI CAO (MANDATORY RULE):
1. TOÀN BỘ NỘI DUNG NGUYÊN VĂN CÁC BƯỚC HƯỚNG DẪN VÀ ĐƯỜNG LINK (https://...) BẮT BUỘC PHẢI TRÍCH XUẤT 100% TỪ KHO DỮ LIỆU BÊN DƯỚI.
2. TUYỆT ĐỐI KHÔNG TỰ BỊA RA CÁC BƯỚC CÀI ĐẶT/XỬ LÝ LỖI KHÔNG CÓ TRONG KHO DỮ LIỆU.
3. TUYỆT ĐỐI KHÔNG TỰ BỊA LINK KHÔNG CÓ TRONG SHEET (như sapo.vn/huong-dan..., youtube.com/watch?v=example...).
4. TUYỆT ĐỐI KHÔNG TỰ Ý CHÈN TỪ "(Ví dụ)" HOẶC "(ví dụ)" VÀO CÂU TRẢ LỜI.
5. Nếu trong Kho dữ liệu KHÔNG CÓ đường link nào, TUYỆT ĐỐI KHÔNG tự tạo dòng dán link hay video.

🎨 QUY TẮC BỐ CỤC TRÌNH BÀY ĐẸP MẮT (PRESENTATION & ICONS RULE):
- Trình bày bài viết thoáng đãng, chia nhỏ dòng rõ ràng.
- Đặt các Icon/Emoji sinh động ở đầu các tiêu đề, từng Cách hoặc từng Bước thao tác (VD: 🛠️, 📌, 💡, 💻, 🍏, 📱, 🎥, 📄, ⚠️, 📞).
- In đậm các từ khóa quan trọng và tên bước thao tác.
- BẮT BUỘC mỗi đường link URL thực tế (https://...) phải nằm trên một dòng riêng biệt bắt đầu bằng dấu gạch đầu dòng (`- `).

🎯 QUY TẮC XỬ LÝ THEO LOẠI THÔNG TIN:

1. 🛑 KHI CHƯA CÓ MODEL MÁY CỤ THỂ (`HAS_DEVICE_INFO: False`):
   - **BẮT BUỘC hỏi lại 1 câu khoanh vùng phân loại máy:**
     "Dạ, để em đưa ra hướng dẫn khắc phục chính xác nhất, anh/chị cho em hỏi mình đang sử dụng loại máy in nào ạ?
      1. 🧾 **Máy in hóa đơn (In bill tính tiền):** Ví dụ SPR02, K200L, K200U, A160M...
      2. 🏷️ **Máy in tem mã vạch (In tem dán sản phẩm):** Ví dụ SPL01, XP-350B, G8...
      
      Anh/chị cho em xin tên model cụ thể ghi trên máy để em gửi hướng dẫn chuẩn 100% cho mình nhé!"

2. 🛑 KHI ĐÃ CÓ MODEL MÁY CỤ THỂ (`HAS_DEVICE_INFO: True`):
   - Trích xuất ĐÚNG NỘI DUNG BÀI HƯỚNG DẪN / CÁCH CÀI ĐẶT trong Kho dữ liệu tương ứng với thiết bị đó.
   - Giữ nguyên các đường link thật (Google Docs, Google Drive, YouTube, Link Driver) nằm trong ô dữ liệu.
   - Nếu người dùng chưa nói rõ dùng Windows hay Mac khi hỏi cài máy tính, chèn câu lịch sự ở đầu:
     "Dạ, em xin gửi anh/chị hướng dẫn cài đặt chi tiết trên máy tính **Windows** ạ. (Nếu mình đang sử dụng máy **Mac / macOS**, anh/chị cứ nhắn em gửi bài hướng dẫn riêng cho Mac nhé! 😊)"

🛑 LUẬT THÉP ĐỊNH DẠNG:
- TUYỆT ĐỐI CẤM sử dụng mã LaTeX toán học (như $\\rightarrow$, $\\Rightarrow$). Dùng dấu mũi tên "➔" hoặc "->".
- Không tự bịa bước Control Panel hay thao tác phần cứng nếu dữ liệu không có.

---

KHO DỮ LIỆU GỐC SAPO:
{knowledge_context}
"""

# ------------------------------------------------------------------------------
# HÀM GỌI LLM CÓ TEMPERATURE = 0.1 ÉP COPY NGUYÊN VĂN LINK
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
# HÀM WRAPPER GỬI TIN NHẮN DÀNH CHO WORKSPACE ADD-ON (GHI CHÚ CHÂN TRANG)
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

    focused_knowledge, has_device_info, top_matches = await get_high_precision_knowledge(req.messages, req.role)
    system_instruction = build_smart_system_prompt(focused_knowledge, has_device_info)

    async def generate_response_stream():
        ans = await call_llm_with_history(system_instruction, req.messages)
        ans = restore_exact_urls(ans, top_matches)
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
            return JSONResponse(content=wrap_gsuite_addon_response("👋 Xin chào! Em là Trợ Lý KHO Sapo. Hãy gõ tên thiết bị hoặc câu hỏi để em hỗ trợ ngay 24/7!", show_reset_note=False))

        clean_user_q = re.sub(r'[^\w\s]', '', cleaned_message.lower()).strip()

        # A. NHẬN DIỆN MỌI CÂU LỆNH CHỮ LÀM MỚI BỘ NHỚ
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

        # B. TỰ ĐỘNG LÀM SẠCH BỘ NHỚ THEO THỜI GIAN RẢNH (TTL DYNAMIC TỪ SHEET)
        now = time.time()
        last_active = GOOGLE_CHAT_LAST_ACTIVE.get(space_id, 0)
        reset_minutes = get_auto_reset_minutes()

        if reset_minutes > 0 and last_active > 0:
            timeout_seconds = reset_minutes * 60
            if (now - last_active) > timeout_seconds:
                GOOGLE_CHAT_HISTORY[space_id] = []

        GOOGLE_CHAT_LAST_ACTIVE[space_id] = now

        # C. LƯU THOẠI VÀ CHẠY RAG CÙNG BỐI CẢNH 4 TẦNG THÔNG MINH
        if space_id not in GOOGLE_CHAT_HISTORY:
            GOOGLE_CHAT_HISTORY[space_id] = []
        
        GOOGLE_CHAT_HISTORY[space_id].append({"role": "user", "text": cleaned_message})
        if len(GOOGLE_CHAT_HISTORY[space_id]) > 10:
            GOOGLE_CHAT_HISTORY[space_id] = GOOGLE_CHAT_HISTORY[space_id][-10:]

        focused_knowledge, has_device_info, top_matches = await get_high_precision_knowledge(GOOGLE_CHAT_HISTORY[space_id], role="Sale")
        system_instruction = build_smart_system_prompt(focused_knowledge, has_device_info)

        ai_response = await call_llm_with_history(system_instruction, GOOGLE_CHAT_HISTORY[space_id])
        ai_response = restore_exact_urls(ai_response, top_matches)

        GOOGLE_CHAT_HISTORY[space_id].append({"role": "assistant", "text": ai_response})

        return JSONResponse(content=wrap_gsuite_addon_response(ai_response, show_reset_note=True))

    except Exception:
        return JSONResponse(content=wrap_gsuite_addon_response("⚠️ Hệ thống AI hiện đang bận xử lý. Anh/chị vui lòng nhắn gửi lại câu hỏi sau vài giây giúp em nhé! 🙏", show_reset_note=False))
