import os
import io
import json
import asyncio
import time
import re
import pandas as pd
import httpx
from datetime import datetime
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Trợ Lý KHO Sapo Synchronized Engine", version="4800.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# CẤU HÌNH BIẾN MÔI TRƯỜNG & RAM CACHE & APPS SCRIPT WEBHOOK
# ------------------------------------------------------------------------------
SHEET_ID = os.getenv("SHEET_ID", "1ZMq0mTiQTDiP92UPaOIv39Q17WJXDiuvrcyYwfs7_Ag").strip()

# ➔ ĐIỀN LINK APPS SCRIPT WEB APP CỦA ANH VÀO ĐÂY HOẶC ĐẶT BIẾN MÔI TRƯỜNG APPS_SCRIPT_URL
APPS_SCRIPT_URL = os.getenv("https://script.google.com/macros/s/AKfycbz0lb86y3O6ynTkGK4yVMSsZrJSB4vKXOoh_cyu4g4JNm2Cr17k7DAQ21l3YoMwLYl9/exec", "").strip()

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "").strip()
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gemma-4-31b").strip()
AVAILABLE_CEREBRAS_MODELS = []

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()

RAM_CACHE = {}
GOOGLE_CHAT_HISTORY = {} 
GOOGLE_CHAT_LAST_ACTIVE = {} 

CHAT_LOGS = []
FEEDBACK_LOGS = []

TABS_PUBLIC = [
    "1_THIET_BI_VA_LOI", 
    "2_HUONG_DAN_CAI_DAT", 
    "3_CHINH_SACH_SAPO", 
    "NHAN_DIEN_THIET_BI"
]
TAB_PRIVATE = "4_DU_LIEU_NOI_BO"
TAB_CONFIG = "0_CAI_DAT"
TAB_LOGS = "5_NHAT_KY_CHAT"
ALL_TABS = [TAB_CONFIG] + TABS_PUBLIC + [TAB_PRIVATE, TAB_LOGS]

HTTP_CLIENT: httpx.AsyncClient = None

# ------------------------------------------------------------------------------
# HÀM ĐẨY LOG VỀ GOOGLE SHEET CHẠY NGẦM (ASYNC BACKGROUND TASK)
# ------------------------------------------------------------------------------
async def send_log_to_google_sheet_async(row_data: list):
    if not APPS_SCRIPT_URL or not HTTP_CLIENT:
        return
    try:
        payload = {
            "tab": TAB_LOGS,
            "row": row_data
        }
        await HTTP_CLIENT.post(APPS_SCRIPT_URL, json=payload, timeout=8.0)
    except Exception as e:
        print(f"⚠️ Lỗi gửi log về Google Sheet: {str(e)}")

# ------------------------------------------------------------------------------
# KHỞI TẠO HTTP CLIENT & NẠP DỮ LIỆU
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
        "engine": "Synchronized Router-AI Engine",
        "has_apps_script_url": bool(APPS_SCRIPT_URL),
        "auto_reset_minutes": get_auto_reset_minutes(),
        "active_cerebras_model": CEREBRAS_MODEL,
        "available_cerebras_models": AVAILABLE_CEREBRAS_MODELS,
        "has_cerebras_key": bool(CEREBRAS_API_KEY),
        "has_gemini_key": bool(GEMINI_API_KEY),
        "total_logs": len(CHAT_LOGS),
        "total_feedbacks": len(FEEDBACK_LOGS)
    }

@app.get("/reload")
async def reload_data():
    return await load_sheet_data_async()

# ------------------------------------------------------------------------------
# LÀM SẠCH VĂN BẢN VÀ KHÔI PHỤC LINK NGUYÊN BẢN TỰ ĐỘNG
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
    if not top_matches:
        return text

    win_url = ""
    mac_url = ""
    doc_url = ""
    video_url = ""
    img_urls = []

    for score, tab, row in top_matches:
        for k, v in row.items():
            val_str = str(v).strip()
            if val_str.startswith("http://") or val_str.startswith("https://"):
                key_lower = str(k).lower()
                if "win" in key_lower: win_url = val_str
                elif "mac" in key_lower: mac_url = val_str
                elif "video" in key_lower or "hd" in key_lower: video_url = val_str
                elif "anh" in key_lower or "img" in key_lower or "truc_tiep" in key_lower: img_urls.append(val_str)
                elif "noi_dung" in key_lower or "huong_dan" in key_lower or "khac_phuc" in key_lower: doc_url = val_str

    url_pattern = re.compile(r'https?://[^\s\)\>\]]+')
    found_urls = url_pattern.findall(text)

    for found_url in found_urls:
        clean_found = found_url.rstrip('.,;')
        if "1S_" in clean_found or "S-S-S" in clean_found or "Driver_Win" in clean_found or "Guide" in clean_found or "link" in clean_found.lower():
            replacement = None
            if "Driver_Win" in clean_found or "win" in clean_found.lower():
                replacement = win_url or doc_url
            elif "Driver_Mac" in clean_found or "mac" in clean_found.lower():
                replacement = mac_url or doc_url
            elif "Video" in clean_found or "youtube" in clean_found.lower() or "hd" in clean_found.lower():
                replacement = video_url
            elif "anh" in clean_found.lower() or "img" in clean_found.lower():
                replacement = img_urls[0] if img_urls else doc_url
            elif "Guide" in clean_found or "doc" in clean_found.lower():
                replacement = doc_url or win_url
            
            if replacement:
                text = text.replace(clean_found, replacement)

    text = re.sub(r'\s*\(\s*Ví dụ\s*\)', '', text, flags=re.IGNORECASE)
    return text

# ------------------------------------------------------------------------------
# TRÍCH XUẤT ĐẦY ĐỦ THÔNG TIN NGƯỜI DÙNG TỪ GOOGLE CHAT WEBHOOK
# ------------------------------------------------------------------------------
def extract_user_text_and_identity(event: dict) -> tuple:
    user_email = ""
    user_name = ""
    
    if isinstance(event.get("user"), dict):
        user_email = event["user"].get("email", user_email)
        user_name = event["user"].get("displayName", user_name)
    
    if isinstance(event.get("message"), dict) and isinstance(event["message"].get("sender"), dict):
        sender = event["message"]["sender"]
        if sender.get("email"): user_email = sender["email"]
        if sender.get("displayName"): user_name = sender["displayName"]

    if isinstance(event.get("chat"), dict) and isinstance(event["chat"].get("messagePayload"), dict):
        m_payload = event["chat"]["messagePayload"]
        if isinstance(m_payload.get("message"), dict) and isinstance(m_payload["message"].get("sender"), dict):
            sender = m_payload["message"]["sender"]
            if sender.get("email"): user_email = sender["email"]
            if sender.get("displayName"): user_name = sender["displayName"]

    user_text = ""
    if isinstance(event.get("message"), dict):
        msg = event["message"]
        if msg.get("text"): user_text = msg["text"]
        elif msg.get("argumentText"): user_text = msg["argumentText"]

    if not user_text and isinstance(event.get("chat"), dict):
        chat = event["chat"]
        if isinstance(chat.get("messagePayload"), dict) and isinstance(chat["messagePayload"].get("message"), dict):
            m = chat["messagePayload"]["message"]
            user_text = m.get("text") or m.get("argumentText") or ""
        elif isinstance(chat.get("message"), dict):
            m = chat["message"]
            user_text = m.get("text") or m.get("argumentText") or ""

    if not user_text:
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
        user_text = deep_search(event)

    clean_name = user_name.strip()
    if clean_name == "Google Chat User":
        clean_name = ""

    return user_text, user_email, clean_name

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

def extract_latest_action_intent_keywords(latest_text: str) -> str:
    txt = latest_text.lower()
    if any(k in txt for k in ["bảo hành", "đổi trả", "chính sách", "thu hồi", "tổng đài", "hotline", "sđt", "liên hệ", "địa chỉ", "kho", "sổ", "admin", "tỉnh"]):
        return "policy"
    if any(k in txt for k in ["lan", "ip", "mạng", "wifi", "app", "xtest", "điện thoại"]):
        return "lan_setup"
    if any(k in txt for k in ["driver", "cài", "setup", "máy tính", "windows", "mac"]):
        return "driver_setup"
    if any(k in txt for k in ["lỗi", "không", "kẹt", "hư", "trắng", "mực", "cắt", "sửa", "ra mực", "không ra", "chậm"]):
        return "error_fix"
    return ""

# ------------------------------------------------------------------------------
# ROUTER AI + FSM MA TRẬN 3 LỚP KẾ THỪA INTENT AN TOÀN 100%
# ------------------------------------------------------------------------------
async def extract_latest_action_intent(messages: list) -> str:
    user_msgs = [m.get("text", "") for m in messages if m.get("role") in ["user", "Khach_Hang"]]
    if not user_msgs:
        return ""
    
    latest_user_text = user_msgs[-1].strip()
    words_in_latest = [w for w in re.sub(r'[^\w\s]', '', latest_user_text).split() if w]
    
    new_intent = ""
    if HTTP_CLIENT and CEREBRAS_API_KEY and CEREBRAS_MODEL and latest_user_text:
        try:
            router_prompt = f"""Bạn là bộ phân loại ý định hỗ trợ kỹ thuật cho Sapo.
Hãy phân loại DUY NHẤT CÂU HỎI MỚI NHẤT của khách hàng vào DUY NHẤT 1 trong các nhãn:
- policy: Bảo hành, đổi trả, số tổng đài, hotline, sđt liên hệ, địa chỉ kho, lịch làm việc, thông tin liên hệ Sapo.
- lan_setup: Cài đặt in qua mạng LAN, địa chỉ IP, kết nối Wifi, in qua App trên điện thoại/tablet (như xTest, Sapo App).
- driver_setup: Tải driver, hướng dẫn cài máy tính Windows, cài Mac/macOS, kết nối USB.
- error_fix: Báo lỗi thiết bị, sự cố kỹ thuật (không in được, kẹt giấy, in ra giấy trắng, không ra mực, kẹt dao cắt, in LAN bị chậm, hỏng hóc, sửa chữa).
- general: Chào hỏi, cảm ơn hoặc chỉ nêu tên model thiết bị không kèm hành động.

Chỉ trả về DUY NHẤT tên nhãn (không viết thêm giải thích).
Câu hỏi: "{latest_user_text}"
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
                        new_intent = valid_intent
                        break
        except Exception:
            pass

    if not new_intent:
        new_intent = extract_latest_action_intent_keywords(latest_user_text)

    if new_intent in ["policy", "lan_setup", "driver_setup", "error_fix"]:
        return new_intent

    # Lớp 2: Kế thừa Intent cho câu trả lời Model ngắn
    if len(words_in_latest) <= 4 and len(messages) >= 2:
        prev_ai_msg = ""
        for m in reversed(messages[:-1]):
            if m.get("role") not in ["user", "Khach_Hang"]:
                prev_ai_msg = m.get("text", "").lower()
                break
        
        if any(kw in prev_ai_msg for kw in ["loại máy", "model", "tên máy", "sử dụng loại", "máy in nào"]):
            if len(user_msgs) >= 2:
                prev_user_text = user_msgs[-2]
                inherited_intent = extract_latest_action_intent_keywords(prev_user_text)
                if inherited_intent in ["lan_setup", "driver_setup", "error_fix", "policy"]:
                    return inherited_intent

    return new_intent or "driver_setup"

# ------------------------------------------------------------------------------
# MA TRẬN ĐIỀU HƯỚNG TRÍCH XUẤT DỮ LIỆU CHÍNH XÁC (HARD FILTER)
# ------------------------------------------------------------------------------
async def get_high_precision_knowledge(messages_list: list, role: str) -> tuple:
    accessible_tabs = ALL_TABS if role == "Sale" else TABS_PUBLIC
    
    user_texts = [m.get("text", "") for m in messages_list if m.get("role") in ["user", "Khach_Hang"]]
    latest_text = user_texts[-1] if user_texts else ""

    detected_model, detected_category = extract_device_info_from_history(messages_list)
    platform_intent = extract_platform_intent(messages_list)
    action_intent = await extract_latest_action_intent(messages_list)
    
    has_device_info = bool(detected_model) or (action_intent == "policy")

    stop_words = {"mình", "có", "bị", "được", "không", "cho", "với", "là", "và", "nhé", "ạ", "cần", "giúp", "tôi", "xin", "lỗi", "thế", "nào", "bao", "nhiêu", "thông", "số", "qua", "đã", "ok", "rồi", "nhưng", "lại", "muốn"}
    words = [w for w in latest_text.lower().split() if len(w) > 1 and w not in stop_words]

    tech_rows = []    
    policy_rows = []  

    for tab in accessible_tabs:
        if tab in [TAB_CONFIG, TAB_LOGS]: continue
        
        if action_intent == "error_fix" and tab == "2_HUONG_DAN_CAI_DAT":
            continue
        if action_intent in ["lan_setup", "driver_setup"] and tab == "1_THIET_BI_VA_LOI":
            continue

        for row in RAM_CACHE.get(tab, []):
            row_text = " ".join(str(v).lower() for v in row.values())
            score = 0
            
            for w in words:
                if w in row_text:
                    score += 25

            if tab in ["3_CHINH_SACH_SAPO", "4_DU_LIEU_NOI_BO"]:
                score += 100  
                if action_intent == "policy":
                    score += 2000
                if score > 0:
                    policy_rows.append((score, tab, row))

            elif tab == "1_THIET_BI_VA_LOI":
                if detected_model and detected_model in row_text: score += 500
                if detected_category and detected_category in row_text: score += 150
                if action_intent == "error_fix": score += 2000
                if score > 0: tech_rows.append((score, tab, row))

            elif tab == "2_HUONG_DAN_CAI_DAT":
                if detected_model and detected_model in row_text: score += 500
                if detected_category and detected_category in row_text: score += 150
                row_thao_tac = str(row.get("Loai_Thao_Tac", "")).lower() + " " + str(row.get("Tu_Khoa_Nhan_Dien", "")).lower()
                if action_intent in ["lan_setup", "driver_setup"]: score += 2000
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
# SYSTEM PROMPT - TỰ ĐỘNG CHUYỂN ĐỔI LINH HOẠT XƯNG HÔ THÂN THIỆN
# ------------------------------------------------------------------------------
def build_smart_system_prompt(knowledge_context: str, has_device_info: bool, user_name: str = "") -> str:
    clean_name = user_name.strip() if user_name else ""
    
    if clean_name and clean_name not in ["Guest_Web", "Google Chat User", "Khach_Hang", "Anh/Chị"]:
        name_rule = f"""👤 THÔNG TIN NGƯỜI ĐANG TRÒ CHUYỆN VỚI BẠN:
- Tên người dùng / Sale: **{clean_name}**
- QUY TẮC XƯNG HÔ BẮT BUỘC: Xưng "Em", gọi trực tiếp bằng tên "**{clean_name}**" hoặc "**anh/chị {clean_name}**" trong câu chào và trong suốt quá trình trả lời để tạo sự thân thiết chuyên nghiệp! (Ví dụ: "Dạ em chào anh {clean_name} ạ!", "Em xin gửi anh {clean_name} hướng dẫn...")."""
    else:
        name_rule = f"""👤 QUY TẮC XƯNG HÔ MẶC ĐỊNH:
- Xưng "Em", gọi người dùng là "**anh/chị**" một cách lịch sự, trang trọng (Ví dụ: "Dạ em chào anh/chị ạ!", "Em xin gửi anh/chị hướng dẫn...")."""

    return f"""
Bạn là **Trợ Lý KHO Sapo** – Trợ lý AI cao cấp, có tư duy logic sâu sắc và am hiểu kỹ thuật thiết bị Sapo.

{name_rule}

🎨 QUY TẮC BỐ CỤC TRÌNH BÀY ĐẸP MẮT & DỄ ĐỌC (PRESENTATION & ICONS RULE):
- **Sử dụng Icon/Emoji sinh động** ở đầu các tiêu đề và từng bước thao tác (VD: 💻, 🍏, 📱, 🛠️, 📌, ✅, 👉, 📄, 🎥, ⚠️, 📞).
- **Chia nhỏ đoạn văn thoáng đãng**, dùng danh sách gạch đầu dòng (Bullet points).
- **In đậm các từ khóa quan trọng**, tên nút bấm, tên bước (VD: **Bước 1: Tải Driver**, **Link Driver Mac:**).

🛠️ QUY TẮC BẮT BUỘC KHI XỬ LÝ LỖI KỸ THUẬT (TAB 1_THIET_BI_VA_LOI):
1. Khi dữ liệu thuộc Tab `1_THIET_BI_VA_LOI`, AI **BẮT BUỘC trích xuất CHÍNH XÁC NGUYÊN VĂN** nội dung hướng dẫn ở cột `Cach_Khac_Phuc`.
2. Dán ĐẦY ĐỦ các đường link hình ảnh minh họa (`Link_Anh_Truc_Tiep`) và video hướng dẫn (`Link_Video_Huong_Dan`) nếu có trong Sheet.
3. TUYỆT ĐỐI KHÔNG tự trả lời lý thuyết chung chung mà bỏ qua các link hình ảnh minh họa thực tế có trong Kho dữ liệu!

🛑 QUY TẮC ÉP TÁCH DÒNG ĐỘC LẬP CHO TẤT CẢ LINK (SEPARATED LINK RULE):
- BẮT BUỘC mỗi đường link URL, hình ảnh hoặc video hướng dẫn phải nằm trên một dòng riêng biệt bắt đầu bằng dấu gạch đầu dòng (`- `).
- TUYỆT ĐỐI CẤM dán 2 icon hay 2 link trên cùng 1 dòng ngang!
- TUYỆT ĐỐI CẤM tự xuất các câu trong ngoặc vuông giả lập như `[Anh/chị vui lòng...]` hay `[Link...]`. Chỉ dán đường link URL thực tế `https://...` nếu có trong Kho dữ liệu.

📌 QUY TẮC BẮT BUỘC VỀ CHÍNH SÁCH BẢO HÀNH & TỔNG ĐÀI (POLICY RULE):
1. Các thông tin nằm trong Tab `3_CHINH_SACH_SAPO` hoặc `4_DU_LIEU_NOI_BO` (như Thời hạn bảo hành, Điều kiện bảo hành, Số tổng đài 1900 6750, Địa chỉ...) là **CHÍNH SÁCH CHUNG ÁP DỤNG CHO TẤT CẢ THIẾT BỊ SAPO** (bao gồm SPR02, SPL01, K200L,...).
2. Khi người dùng hỏi về Bảo hành hay Tổng đài hỗ trợ cho một thiết bị cụ thể (VD: SPR02), AI **BẮT BUỘC sử dụng ngay thông tin chính sách chung trong Kho dữ liệu để trả lời trực tiếp**.
3. **TUYỆT ĐỐI CẤM BÁO:** "Không có văn bản cụ thể về chính sách bảo hành cho từng thiết bị" hoặc "Không có thông tin chi tiết về số tổng đài".

🎯 BỘ QUY TẮC XỬ LÝ QUAN TRỌNG NHẤT (TUÂN THỦ 100%):

1. 🛑 QUY TẮC PHÁT HIỆN THIẾU LOẠI MÁY IN / MODEL MÁY (DEVICE CLARIFICATION RULE):
   - Nếu trong Kho dữ liệu báo `HAS_DEVICE_INFO: False` (nghĩa là người dùng CHƯA CUNG CẤP tên model máy cụ thể):
   - **TUYỆT ĐỐI CẤM tự ý suy đoán người dùng đang dùng Máy in hóa đơn hay Máy in tem!**
   - **TUYỆT ĐỐI CẤM tự ý xả bài hướng dẫn xử lý lỗi hay bài cài đặt của Máy in hóa đơn!**
   - **BẮT BUỘC hỏi lại ngay 1 câu khoanh vùng phân loại máy:**
     "Dạ, để em đưa ra hướng dẫn khắc phục chính xác nhất, anh/chị cho em hỏi mình đang sử dụng loại máy in nào ạ?
      1. 🧾 **Máy in hóa đơn (In bill tính tiền):** Ví dụ SPR02, K200L, K200U, A160M...
      2. 🏷️ **Máy in tem mã vạch (In tem dán sản phẩm):** Ví dụ SPL01, XP-350B, G8...
      
      Anh/chị cho em xin tên model cụ thể ghi trên máy để em gửi hướng dẫn chuẩn 100% cho mình nhé!"

2. 🛑 TRƯỜNG HỢP 2: KHI ĐÃ CÓ TÊN MODEL MÁY CỤ THỂ (`HAS_DEVICE_INFO: True`):
   - **BÁM SÁT 100% NỘI DUNG SHEET:** BẮT BUỘC phải trích xuất chính xác từng câu, từng bước và link có trong Dữ liệu bên dưới.
   - Khi Dữ liệu có bài hướng dẫn cài trên **Mac** (như Dòng 3 Tab 2), AI BẮT BUỘC xuất đầy đủ bài hướng dẫn Mac và dán link `Link_Driver_Mac`. TUYỆT ĐỐI KHÔNG báo "không có dữ liệu Mac"!
   - Khi Dữ liệu có bài hướng dẫn cài trên **Windows**, AI xuất bài Windows và link `Link_Driver_Win`.

   👉 💡 QUY TẮC MỞ ĐẦU LỊCH SỰ KHI GỬI BÀI WINDOWS MẶC ĐỊNH:
      - Khi người dùng chỉ nói chung chung "mới đổi máy tính" hoặc "máy tính" (mà CHƯA NÓI RÕ là Windows hay Mac), khi gửi bài Windows, AI BẮT BUỘC chèn 1 câu dẫn dắt tự nhiên ở đầu:
        "Dạ em xin gửi anh/chị hướng dẫn cài đặt chi tiết trên máy tính **Windows** ạ. (Nếu mình đang sử dụng máy **Mac / macOS**, anh/chị cứ nhắn em gửi bài hướng dẫn riêng cho Mac nhé! 😊)"

👉 🛑 QUY TẮC ĐÍNH KÈM LINK VÀ NỘI DUNG SHEET (TUÂN THỦ BẮT BUỘC):
   - BẮT BUỘC Copy chính xác 100% từng ký tự URL từ Kho dữ liệu, CẤM CẮT NGẮN, CẤM VIẾT TẮT (`...`) HOẶC TỰ BỊA LINK KHÔNG CÓ TRONG SHEET.
   - Xuất đầy đủ tất cả các link nếu có trong dòng dữ liệu (TUYỆT ĐỐI KHÔNG BỎ BỚT LINK VIDEO HOẶC LINK TÀI LIỆU):
     - 📄 **Tài liệu / Bài viết hướng dẫn:** <Link trong Sheet>
     - 💻 **Link Tải Driver Windows:** <Link_Driver_Win trong Sheet>
     - 🍏 **Link Tải Driver Mac:** <Link_Driver_Mac trong Sheet>
     - 🎥 **Video Hướng Dẫn:** <Link_Video_Huong_Dan trong Sheet>
     - 🖼️ **Hình ảnh minh họa:** <Link_Anh_Truc_Tiep trong Sheet>
     - ⚠️ **Lưu ý quan trọng:** <Luu_y trong Sheet>

🧠 QUY TẮC DUY TRÌ BỐI CẢNH (Context Memory):
   - Khi người dùng đã cung cấp thông tin (như Hệ điều hành Mac/Win hoặc tên Model), BẮT BUỘC phải giữ nguyên bối cảnh đó suốt cả cuộc hội thoại trừ khi người dùng chủ động yêu cầu đổi!

🛑 LUẬT THÉP ĐỊNH DẠNG:
- TUYỆT ĐỐI CẤM sử dụng mã LaTeX toán học (như $\\rightarrow$, $\\Rightarrow$). Dùng dấu mũi tên "➔" hoặc "->".
- Không tự bịa bước Control Panel hay thao tác phần cứng nếu dữ liệu không có.
- Chỉ cung cấp đường link có thực trong Kho dữ liệu dưới đây.

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
# HÀM WRAPPER GỬI TIN NHẮN DÀNH CHO WORKSPACE ADD-ON
# ------------------------------------------------------------------------------
def wrap_gsuite_addon_response(text_message: str, show_reset_note: bool = True) -> dict:
    clean_text = clean_thinking_process(text_message)
    clean_text = re.sub(r'\[(.*?)\]\((https?://.*?)\)', r'\1 (\2)', clean_text)
    clean_text = re.sub(r'\*{2,3}', '*', clean_text)
    
    if show_reset_note and "Em đã xóa bộ nhớ" not in clean_text:
        footer_note = (
            "\n\n───────────────────────────────\n"
            " 💡 _Mẹo: Bạn có thể nhắn \"xóa lịch sử\" hoặc \"bắt đầu cuộc trò chuyện mới\" khi cần hỏi sang thiết bị/chủ đề khác nhé!_"
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
# XÁC THỰC MẬT KHẨU SALE
# ------------------------------------------------------------------------------
def get_valid_sale_passcodes() -> list:
    valid_codes = ["sapo123", "sapo2026", "sapo@2026", "123456"]
    cai_dat_records = RAM_CACHE.get(TAB_CONFIG, [])
    if cai_dat_records:
        for row in cai_dat_records:
            vals = [str(v).strip() for v in row.values() if str(v).strip()]
            row_text = " ".join(vals).lower()
            if any(kw in row_text for kw in ["mat_khau", "passcode", "pin", "sale"]):
                for v in vals:
                    if v.lower() not in ["mat_khau_sale", "mat_khau", "passcode", "0_cai_dat"]:
                        valid_codes.append(v)
    return [c.strip().lower() for c in valid_codes if c.strip()]

class VerifySaleRequest(BaseModel):
    email: str
    passcode: str

@app.post("/verify-sale")
async def verify_sale(req: VerifySaleRequest):
    if not req.email.lower().strip().endswith("@sapo.vn"):
        return {"success": False, "message": "Email bắt buộc phải có đuôi @sapo.vn!"}
    
    allowed_passcodes = get_valid_sale_passcodes()
    user_input_pass = req.passcode.strip().lower()
    
    if user_input_pass in allowed_passcodes:
        return {
            "success": True, 
            "message": f"Xác thực thành công! Xin chào Sale Sapo ({req.email})."
        }
    else:
        return {
            "success": False, 
            "message": "Mật khẩu Sale nội bộ không chính xác!"
        }

# ------------------------------------------------------------------------------
# API ĐÁNH GIÁ PHẢN HỒI (FEEDBACK & LOGGING VỀ GOOGLE SHEET)
# ------------------------------------------------------------------------------
class FeedbackRequest(BaseModel):
    msg_id: str
    rating: str
    user_id: str = "Khach_Hang"
    email: str = ""
    user_question: str = ""
    ai_response: str = ""
    comment: str = ""
    phone: str = ""

@app.post("/feedback")
async def save_feedback(req: FeedbackRequest, background_tasks: BackgroundTasks):
    now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    rating_str = "👍 Hữu ích" if req.rating == "like" else "👎 Không hữu ích"
    
    log_entry = {
        "timestamp": now_str,
        "msg_id": req.msg_id,
        "rating": rating_str,
        "user_id": req.user_id,
        "email": req.email,
        "phone": req.phone,
        "question": req.user_question,
        "response": req.ai_response[:200] + "..." if len(req.ai_response) > 200 else req.ai_response,
        "comment": req.comment
    }
    FEEDBACK_LOGS.append(log_entry)
    
    sheet_row = [
        now_str,
        "Web Feedback",
        req.user_id,
        req.email or req.phone or "N/A",
        "Khách/Sale",
        req.user_question,
        f"[{rating_str}] " + (req.ai_response[:100] + "..."),
        req.comment
    ]
    background_tasks.add_task(send_log_to_google_sheet_async, sheet_row)

    return {"status": "success", "message": "Cảm ơn bạn đã đóng góp ý kiến!"}

@app.get("/analytics")
def get_analytics():
    return {
        "total_chat_logs": len(CHAT_LOGS),
        "total_feedbacks": len(FEEDBACK_LOGS),
        "recent_logs": CHAT_LOGS[-20:],
        "recent_feedbacks": FEEDBACK_LOGS[-20:]
    }

# ------------------------------------------------------------------------------
# 1. CỔNG WEB CHAT (/chat)
# ------------------------------------------------------------------------------
class ChatRequest(BaseModel):
    messages: list
    role: str = "Khach_Hang"
    user_id: str = "Guest_Web"
    user_email: str = ""

@app.post("/chat")
async def chat_stream(req: ChatRequest, background_tasks: BackgroundTasks):
    latest_msg = req.messages[-1]["text"] if req.messages else ""
    clean_q = re.sub(r'[^\w\s]', '', latest_msg.lower()).strip()
    
    exact_quick_greetings = {"chào", "chào bạn", "chào bjan", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em", "chao ban nhe", "xin chào"}
    if clean_q in exact_quick_greetings:
        async def greeting_gen():
            yield "Xin chào anh/chị! Em là **Trợ Lý KHO Sapo**. Anh/chị cần hỗ trợ tra cứu thông số thiết bị hay cài đặt máy in nào ạ?"
        return StreamingResponse(greeting_gen(), media_type="text/plain")

    focused_knowledge, has_device_info, top_matches = await get_high_precision_knowledge(req.messages, req.role)
    system_instruction = build_smart_system_prompt(focused_knowledge, has_device_info, user_name="")

    async def generate_response_stream():
        ans = await call_llm_with_history(system_instruction, req.messages)
        ans = restore_exact_urls(ans, top_matches)
        
        now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        CHAT_LOGS.append({
            "timestamp": now_str,
            "channel": "Web Chat",
            "user_id": req.user_id,
            "email": req.user_email,
            "role": req.role,
            "question": latest_msg,
            "answer_preview": ans[:150]
        })
        
        sheet_row = [
            now_str,
            "Web Chat",
            req.user_id,
            req.user_email or "Guest Web",
            req.role,
            latest_msg,
            ans[:300] + "...",
            ""
        ]
        background_tasks.add_task(send_log_to_google_sheet_async, sheet_row)
        
        yield ans

    return StreamingResponse(generate_response_stream(), media_type="text/plain")

# ------------------------------------------------------------------------------
# 2. CỔNG GOOGLE CHAT BOT (/google-chat) - LẤY TÊN VÀ EMAIL CHÍNH XÁC
# ------------------------------------------------------------------------------
@app.post("/google-chat")
async def google_chat_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        event = await request.json()
        user_message, user_email, user_name = extract_user_text_and_identity(event)
        cleaned_message = re.sub(r'<.*?>', '', user_message).replace("@Trợ Lý KHO Sapo", "").strip()

        space_id = event.get("space", {}).get("name") or event.get("user", {}).get("name") or "default_space"

        event_type = event.get("type") or event.get("chat", {}).get("type") or ""
        if event_type == "ADDED_TO_SPACE":
            greeting_name = f" **{user_name}**" if user_name else ""
            return JSONResponse(content=wrap_gsuite_addon_response(f"👋 Dạ em chào anh/chị{greeting_name}! Em là Trợ Lý KHO Sapo. Hãy gõ tên thiết bị hoặc câu hỏi để em hỗ trợ ngay 24/7!", show_reset_note=False))

        clean_user_q = re.sub(r'[^\w\s]', '', cleaned_message.lower()).strip()

        reset_keywords = {
            "xóa lịch sử", "xoa lich su", "bắt đầu lại", "bat dau lai", 
            "hỏi máy khác", "hoi may khac","chủ đề mới", "chu de moi",
            "bắt đầu cuộc trò chuyện mới", "bat dau cuoc tro chuyen moi",
            "bắt đầu trò chuyện mới", "bat dau tro chuyen moi",
            "xóa lịch sử cuộc trò chuyện", "xoa lich su cuoc tro chuyen"
        }
        if clean_user_q in reset_keywords:
            GOOGLE_CHAT_HISTORY[space_id] = []
            GOOGLE_CHAT_LAST_ACTIVE[space_id] = time.time()
            greeting_name = f" **{user_name}**" if user_name else ""
            return JSONResponse(content=wrap_gsuite_addon_response(f"🧹 Dạ em chào anh/chị{greeting_name}, em đã xóa bộ nhớ bối cảnh cuộc trò chuyện! Anh/chị cần em hỗ trợ cài đặt hay xử lý lỗi thiết bị nào mới ạ? 😊", show_reset_note=False))

        exact_quick_greetings = {"chào", "chào bạn", "chào bjan", "hi", "hello", "chaof bạn", "chao ban", "alo", "chào em", "chao ban nhe", "xin chào"}
        if not cleaned_message or clean_user_q in exact_quick_greetings:
            greeting_name = f" **{user_name}**" if user_name else ""
            return JSONResponse(content=wrap_gsuite_addon_response(f"👋 Dạ em chào anh/chị{greeting_name}! Em là Trợ Lý KHO Sapo. Anh/chị cần hỗ trợ tra cứu thông số máy in hay cài đặt thiết bị nào ạ?", show_reset_note=False))

        now = time.time()
        last_active = GOOGLE_CHAT_LAST_ACTIVE.get(space_id, 0)
        reset_minutes = get_auto_reset_minutes()

        if reset_minutes > 0 and last_active > 0:
            timeout_seconds = reset_minutes * 60
            if (now - last_active) > timeout_seconds:
                GOOGLE_CHAT_HISTORY[space_id] = []

        GOOGLE_CHAT_LAST_ACTIVE[space_id] = now

        if space_id not in GOOGLE_CHAT_HISTORY:
            GOOGLE_CHAT_HISTORY[space_id] = []
        
        GOOGLE_CHAT_HISTORY[space_id].append({"role": "user", "text": cleaned_message})
        if len(GOOGLE_CHAT_HISTORY[space_id]) > 10:
            GOOGLE_CHAT_HISTORY[space_id] = GOOGLE_CHAT_HISTORY[space_id][-10:]

        focused_knowledge, has_device_info, top_matches = await get_high_precision_knowledge(GOOGLE_CHAT_HISTORY[space_id], role="Sale")
        
        system_instruction = build_smart_system_prompt(focused_knowledge, has_device_info, user_name=user_name)

        ai_response = await call_llm_with_history(system_instruction, GOOGLE_CHAT_HISTORY[space_id])
        ai_response = restore_exact_urls(ai_response, top_matches)

        GOOGLE_CHAT_HISTORY[space_id].append({"role": "assistant", "text": ai_response})

        now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        CHAT_LOGS.append({
            "timestamp": now_str,
            "channel": "Google Chat",
            "user_id": user_name or "Workspace User",
            "email": user_email or "Workspace User",
            "role": "Sale",
            "question": cleaned_message,
            "answer_preview": ai_response[:150]
        })

        sheet_row = [
            now_str,
            "Google Chat",
            user_name or "Workspace User",
            user_email or "Workspace User",
            "Sale Nội Bộ",
            cleaned_message,
            ai_response[:300] + "...",
            ""
        ]
        background_tasks.add_task(send_log_to_google_sheet_async, sheet_row)

        return JSONResponse(content=wrap_gsuite_addon_response(ai_response, show_reset_note=True))

    except Exception:
        return JSONResponse(content=wrap_gsuite_addon_response("⚠️ Hệ thống AI hiện đang bận xử lý. Anh/chị vui lòng nhắn gửi lại câu hỏi sau vài giây giúp em nhé! 🙏", show_reset_note=False))
