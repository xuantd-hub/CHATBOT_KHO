import os
import io
import json
import asyncio
import re
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ==============================================================================
# TRỢ LÝ KHO SAPO - VERSION 500
# Kiến trúc:
# 1) Python xử lý state / menu / nhận diện thiết bị / intent cơ bản
# 2) Google Sheet là nguồn dữ liệu sự thật
# 3) LLM chỉ dùng để hiểu câu hỏi và diễn đạt câu trả lời
# 4) Không cho LLM tự bịa link, model, thông số
# ==============================================================================

APP_VERSION = "500.0"

app = FastAPI(
    title="Trợ Lý KHO Sapo - Smart Router & Knowledge Engine",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# CONFIG
# ==============================================================================

SHEET_ID = os.getenv(
    "SHEET_ID",
    "1ZMq0mTiQTDiP92UPaOIv39Q17WJXDiuvrcyYwfs7_Ag",
).strip()

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "").strip()
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b").strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()

# Cache dữ liệu Google Sheet trong RAM.
RAM_CACHE: Dict[str, List[Dict[str, str]]] = {}

# Session Google Chat.
# space_id -> {
#   device: tên thiết bị,
#   intent: intent gần nhất,
#   updated_at: timestamp,
#   history: các lượt gần nhất
# }
GOOGLE_CHAT_SESSION_CACHE: Dict[str, Dict[str, Any]] = {}

AVAILABLE_CEREBRAS_MODELS: List[str] = []
HTTP_CLIENT: Optional[httpx.AsyncClient] = None

TABS_PUBLIC = [
    "1_THIET_BI_VA_LOI",
    "2_HUONG_DAN_CAI_DAT",
    "3_CHINH_SACH_SAPO",
    "NHAN_DIEN_THIET_BI",
]

TAB_PRIVATE = "4_DU_LIEU_NOI_BO"
ALL_TABS = TABS_PUBLIC + [TAB_PRIVATE]


# ==============================================================================
# FASTAPI MODELS
# ==============================================================================

class ChatRequest(BaseModel):
    messages: list
    role: str = "Khach_Hang"


# ==============================================================================
# STARTUP / SHUTDOWN
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    global HTTP_CLIENT

    HTTP_CLIENT = httpx.AsyncClient(
        timeout=httpx.Timeout(15.0, read=45.0),
        limits=httpx.Limits(
            max_keepalive_connections=30,
            max_connections=100,
        ),
    )

    # Quan trọng: chờ load Sheet xong trước khi nhận request.
    await load_sheet_data_async()
    await discover_active_cerebras_models()


@app.on_event("shutdown")
async def shutdown_event():
    global HTTP_CLIENT

    if HTTP_CLIENT:
        await HTTP_CLIENT.aclose()
        HTTP_CLIENT = None


# ==============================================================================
# CEREBRAS MODEL DISCOVERY
# ==============================================================================

async def discover_active_cerebras_models():
    global CEREBRAS_MODEL, AVAILABLE_CEREBRAS_MODELS

    if not CEREBRAS_API_KEY or not HTTP_CLIENT:
        return

    url = "https://api.cerebras.ai/v1/models"
    headers = {
        "Authorization": f"Bearer {CEREBRAS_API_KEY}",
    }

    try:
        res = await HTTP_CLIENT.get(
            url,
            headers=headers,
            timeout=6.0,
        )

        if res.status_code == 200:
            models_data = res.json().get("data", [])
            model_ids = [
                str(m.get("id", "")).strip()
                for m in models_data
                if m.get("id")
            ]

            AVAILABLE_CEREBRAS_MODELS = model_ids

            # Ưu tiên model mà user đã cấu hình nếu nó thực sự tồn tại.
            if CEREBRAS_MODEL in model_ids:
                return

            preferred = [
                "gpt-oss-120b",
                "llama-4-scout-17b-16e-instruct",
                "llama3.1-70b",
            ]

            for model in preferred:
                if model in model_ids:
                    CEREBRAS_MODEL = model
                    return

            if model_ids:
                CEREBRAS_MODEL = model_ids[0]

    except Exception:
        # Không làm app chết chỉ vì API /models lỗi.
        pass


# ==============================================================================
# GOOGLE SHEET
# ==============================================================================

async def fetch_single_tab_raw(tab: str) -> Tuple[str, List[Dict[str, str]]]:
    if not HTTP_CLIENT:
        return tab, []

    url = (
        f"https://docs.google.com/spreadsheets/d/{SHEET_ID}"
        f"/gviz/tq?tqx=out:csv&sheet={tab}"
    )

    try:
        res = await HTTP_CLIENT.get(url, timeout=10.0)

        if res.status_code != 200:
            return tab, []

        content_type = res.headers.get("Content-Type", "")
        if "text/csv" not in content_type and not res.content:
            return tab, []

        df = pd.read_csv(io.BytesIO(res.content)).fillna("")

        records: List[Dict[str, str]] = []

        for _, row in df.iterrows():
            row_data: Dict[str, str] = {}

            for key, value in row.items():
                key_text = str(key).strip()
                value_text = str(value).strip()

                if key_text and value_text and value_text.lower() != "nan":
                    row_data[key_text] = value_text

            if row_data:
                records.append(row_data)

        return tab, records

    except Exception:
        return tab, []


async def load_sheet_data_async():
    global RAM_CACHE

    tasks = [
        fetch_single_tab_raw(tab)
        for tab in ALL_TABS
    ]

    results = await asyncio.gather(*tasks)

    RAM_CACHE = {
        tab: records
        for tab, records in results
    }

    return {
        "status": "success",
        "tabs": {
            tab: len(records)
            for tab, records in RAM_CACHE.items()
        },
    }


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "version": APP_VERSION,
        "active_cerebras_model": CEREBRAS_MODEL,
        "available_cerebras_models": AVAILABLE_CEREBRAS_MODELS,
        "has_cerebras_key": bool(CEREBRAS_API_KEY),
        "has_gemini_key": bool(GEMINI_API_KEY),
        "loaded_rows": {
            tab: len(records)
            for tab, records in RAM_CACHE.items()
        },
    }


@app.get("/reload")
async def reload_data():
    return await load_sheet_data_async()


# ==============================================================================
# TEXT NORMALIZATION
# ==============================================================================

def normalize_text(text: str) -> str:
    """
    Chuẩn hóa tiếng Việt để tìm kiếm:
    - lowercase
    - bỏ dấu
    - chuẩn hóa khoảng trắng
    - giữ chữ/số
    """
    if not text:
        return ""

    text = str(text).lower().strip()

    text = unicodedata.normalize("NFD", text)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) != "Mn"
    )

    text = text.replace("đ", "d")

    text = re.sub(r"[^a-z0-9\s._/-]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_google_chat_text(text: str) -> str:
    if not text:
        return ""

    # Xóa HTML đơn giản.
    text = re.sub(r"<[^>]+>", " ", text)

    # Xóa mention bot.
    text = re.sub(
        r"@Trợ\s*Lý\s*KHO\s*Sapo",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==============================================================================
# GOOGLE CHAT EVENT EXTRACTION
# ==============================================================================

def extract_user_text(event: dict) -> str:
    if not isinstance(event, dict):
        return ""

    if isinstance(event.get("message"), dict):
        msg = event["message"]

        if msg.get("text"):
            return str(msg["text"])

        if msg.get("argumentText"):
            return str(msg["argumentText"])

    if isinstance(event.get("chat"), dict):
        chat = event["chat"]

        payload = chat.get("messagePayload")

        if isinstance(payload, dict):
            msg = payload.get("message")

            if isinstance(msg, dict):
                if msg.get("text"):
                    return str(msg["text"])

                if msg.get("argumentText"):
                    return str(msg["argumentText"])

        msg = chat.get("message")

        if isinstance(msg, dict):
            if msg.get("text"):
                return str(msg["text"])

            if msg.get("argumentText"):
                return str(msg["argumentText"])

    def deep_search(obj):
        if isinstance(obj, dict):
            if (
                isinstance(obj.get("argumentText"), str)
                and obj["argumentText"].strip()
            ):
                return obj["argumentText"]

            if (
                isinstance(obj.get("text"), str)
                and obj["text"].strip()
                and not obj["text"].startswith("spaces/")
            ):
                return obj["text"]

            for value in obj.values():
                result = deep_search(value)

                if result:
                    return result

        elif isinstance(obj, list):
            for item in obj:
                result = deep_search(item)

                if result:
                    return result

        return ""

    return deep_search(event)


# ==============================================================================
# GREETING / MENU
# ==============================================================================

GREETING_WORDS = {
    "chao",
    "chao ban",
    "chao em",
    "hi",
    "hello",
    "alo",
    "xin chao",
    "chao ban nhe",
}


def is_greeting(text: str) -> bool:
    normalized = normalize_text(text)

    if normalized in GREETING_WORDS:
        return True

    # Chỉ coi là greeting nếu câu rất ngắn.
    if len(normalized.split()) <= 4 and normalized.startswith("chao "):
        return True

    return False


def normalize_menu_choice(text: str) -> Optional[str]:
    """
    Menu cố định:
    1 = Driver máy tính
    2 = Điện thoại / POS / LAN
    3 = Khắc phục sự cố
    """
    normalized = normalize_text(text).strip()

    if normalized in {"1", "1.", "mot", "một"}:
        return "driver"

    if normalized in {"2", "2.", "hai"}:
        return "mobile"

    if normalized in {"3", "3.", "ba"}:
        return "troubleshoot"

    return None


MENU_TEXT = (
    "Dạ thiết bị **{device}**, anh/chị đang cần em hỗ trợ mục nào dưới đây ạ?\n\n"
    "1. 💻 **Cài Driver trên máy tính** (Windows / macOS)\n"
    "2. 📱 **Cài đặt trên điện thoại / máy POS** "
    "(App XTEST / LAN / đổi IP)\n"
    "3. 🛠️ **Khắc phục sự cố** "
    "(không cắt giấy, giấy trắng, nghẽn mạng, báo đèn đỏ...)\n\n"
    "Anh/chị chỉ cần trả lời **1, 2 hoặc 3**, em xử lý tiếp ngay ạ."
)


# ==============================================================================
# DEVICE DETECTION
# ==============================================================================

DEVICE_KEYS = [
    "Ten_Thiet_Bi",
    "Tên thiết bị",
    "Loai_Thiet_Bi",
    "Loại thiết bị",
    "Model",
    "MODEL",
    "Device",
    "device",
]


def get_device_name(row: Dict[str, str]) -> str:
    for key in DEVICE_KEYS:
        value = str(row.get(key, "")).strip()

        if value:
            return value

    return ""


def all_known_devices(role: str = "Khach_Hang") -> List[str]:
    tabs = ALL_TABS if role == "Sale" else TABS_PUBLIC

    names = set()

    for tab in tabs:
        for row in RAM_CACHE.get(tab, []):
            name = get_device_name(row)

            if name:
                names.add(name)

    # Thiết bị dài hơn đứng trước để tránh match "K200" trước "K200L".
    return sorted(
        names,
        key=lambda x: len(normalize_text(x)),
        reverse=True,
    )


def find_device_in_query(
    query: str,
    role: str = "Khach_Hang",
) -> Optional[str]:
    normalized_query = normalize_text(query)

    if not normalized_query:
        return None

    devices = all_known_devices(role)

    # 1. Match chính xác tên thiết bị trong câu.
    for device in devices:
        normalized_device = normalize_text(device)

        if not normalized_device:
            continue

        if (
            normalized_query == normalized_device
            or re.search(
                rf"(?<![a-z0-9]){re.escape(normalized_device)}"
                rf"(?![a-z0-9])",
                normalized_query,
            )
        ):
            return device

    # 2. Match dạng substring cho model có ký tự đặc biệt.
    compact_query = normalized_query.replace(" ", "")

    for device in devices:
        compact_device = normalize_text(device).replace(" ", "")

        if len(compact_device) >= 3 and compact_device in compact_query:
            return device

    return None


# ==============================================================================
# INTENT DETECTION - DETERMINISTIC FIRST
# ==============================================================================

def detect_intent(text: str) -> Optional[str]:
    q = normalize_text(text)

    if not q:
        return None

    # Driver
    driver_patterns = [
        "driver",
        "cai driver",
        "cai may in",
        "cai tren may tinh",
        "may tinh",
        "windows",
        "win 10",
        "win 11",
        "macos",
        "mac",
        "download driver",
        "tai driver",
    ]

    if any(p in q for p in driver_patterns):
        return "driver"

    # Mobile / POS / LAN
    mobile_patterns = [
        "dien thoai",
        "may pos",
        "pos",
        "xtest",
        "ket noi lan",
        "ket noi wifi",
        "doi ip",
        "ip may in",
        "in qua dien thoai",
        "in tren dien thoai",
        "android",
        "iphone",
    ]

    if any(p in q for p in mobile_patterns):
        return "mobile"

    # Troubleshooting
    trouble_patterns = [
        "loi",
        "khong in",
        "khong nhan",
        "khong ket noi",
        "in trang",
        "giay trang",
        "khong cat",
        "cat giay",
        "nghen mang",
        "bao den do",
        "den do",
        "khong ra giay",
        "in sai",
        "in bi mo",
        "in cham",
        "in cham",
        "mat ket noi",
        "khong phat hien",
        "khong hoat dong",
        "bi treo",
        "bi dung",
        "su co",
    ]

    if any(p in q for p in trouble_patterns):
        return "troubleshoot"

    # Tra cứu thông tin thiết bị.
    info_patterns = [
        "thong so",
        "thong tin",
        "model",
        "kich thuoc",
        "cong ket noi",
        "usb",
        "lan",
        "bluetooth",
        "bao nhieu",
    ]

    if any(p in q for p in info_patterns):
        return "info"

    return None


def build_intent_query(
    device: Optional[str],
    intent: Optional[str],
    original_message: str,
) -> str:
    if intent == "driver":
        return (
            f"Thiết bị: {device or 'chưa xác định'}\n"
            "Nhu cầu: Cài Driver trên máy tính Windows/macOS\n"
            f"Câu hỏi gốc: {original_message}"
        )

    if intent == "mobile":
        return (
            f"Thiết bị: {device or 'chưa xác định'}\n"
            "Nhu cầu: Cài đặt trên điện thoại / POS / LAN / đổi IP\n"
            f"Câu hỏi gốc: {original_message}"
        )

    if intent == "troubleshoot":
        return (
            f"Thiết bị: {device or 'chưa xác định'}\n"
            "Nhu cầu: Khắc phục sự cố\n"
            f"Câu hỏi gốc: {original_message}"
        )

    return (
        f"Thiết bị: {device or 'chưa xác định'}\n"
        f"Câu hỏi: {original_message}"
    )


# ==============================================================================
# SESSION
# ==============================================================================

SESSION_TTL_SECONDS = 60 * 60 * 8
MAX_SESSION_HISTORY = 8


def get_session(space_id: str) -> Dict[str, Any]:
    session = GOOGLE_CHAT_SESSION_CACHE.get(space_id)

    if not session:
        session = {
            "device": None,
            "intent": None,
            "updated_at": time.time(),
            "history": [],
        }
        GOOGLE_CHAT_SESSION_CACHE[space_id] = session
        return session

    if time.time() - float(session.get("updated_at", 0)) > SESSION_TTL_SECONDS:
        session = {
            "device": None,
            "intent": None,
            "updated_at": time.time(),
            "history": [],
        }
        GOOGLE_CHAT_SESSION_CACHE[space_id] = session

    return session


def update_session(
    space_id: str,
    device: Optional[str] = None,
    intent: Optional[str] = None,
    user_message: Optional[str] = None,
    assistant_message: Optional[str] = None,
):
    session = get_session(space_id)

    if device:
        session["device"] = device

    if intent:
        session["intent"] = intent

    session["updated_at"] = time.time()

    history = session.setdefault("history", [])

    if user_message:
        history.append({
            "role": "user",
            "text": user_message,
        })

    if assistant_message:
        history.append({
            "role": "assistant",
            "text": assistant_message,
        })

    session["history"] = history[-MAX_SESSION_HISTORY:]


# ==============================================================================
# KNOWLEDGE RETRIEVAL
# ==============================================================================

STOP_WORDS = {
    "mình", "minh", "có", "co", "bị", "bi", "được", "duoc",
    "không", "khong", "cho", "với", "voi", "là", "la", "và", "va",
    "nhé", "nhe", "ạ", "a", "cần", "can", "giúp", "giup", "tôi", "toi",
    "xin", "lỗi", "loi", "máy", "may", "thế", "the", "nào", "nao",
    "bao", "nhiêu", "nhieu", "thông", "thong", "số", "so", "in",
    "qua", "đã", "da", "ok", "em", "anh", "chị", "chi",
    "của", "cua", "cho", "một", "mot",
}


def tokenize(text: str) -> List[str]:
    normalized = normalize_text(text)

    tokens = [
        token
        for token in normalized.split()
        if len(token) >= 2 and token not in STOP_WORDS
    ]

    return tokens


def row_text(row: Dict[str, str]) -> str:
    return " ".join(
        str(value)
        for value in row.values()
        if str(value).strip()
    )


def score_row(
    query: str,
    row: Dict[str, str],
    device: Optional[str],
    intent: Optional[str],
) -> float:
    q_norm = normalize_text(query)
    q_tokens = tokenize(query)

    text = normalize_text(row_text(row))
    device_name = normalize_text(get_device_name(row))

    score = 0.0

    # Thiết bị là tín hiệu mạnh nhất.
    if device and device_name:
        d = normalize_text(device)

        if device_name == d:
            score += 500
        elif d in device_name or device_name in d:
            score += 300

    # Token khớp trong tên thiết bị.
    for token in q_tokens:
        if token in device_name:
            score += 80
        elif token in text:
            score += 5

    # Cả cụm query xuất hiện.
    if q_norm and q_norm in text:
        score += 30

    # Intent được ưu tiên bằng tên cột / nội dung.
    intent_keywords = {
        "driver": [
            "driver", "cai dat", "windows", "mac", "macos",
            "download", "tai",
        ],
        "mobile": [
            "dien thoai", "pos", "xtest", "lan", "ip", "android",
            "iphone",
        ],
        "troubleshoot": [
            "loi", "khong in", "giay trang", "khong cat",
            "nghen mang", "den do", "khong ket noi", "su co",
        ],
        "info": [
            "thong so", "model", "usb", "lan", "bluetooth",
        ],
    }

    for keyword in intent_keywords.get(intent or "", []):
        if normalize_text(keyword) in text:
            score += 15

    return score


def get_high_precision_knowledge(
    query: str,
    role: str,
    device: Optional[str] = None,
    intent: Optional[str] = None,
    max_rows: int = 8,
) -> str:
    """
    Retrieval 2 tầng:
    1. Nếu biết thiết bị -> ưu tiên tuyệt đối các row của thiết bị.
    2. Trong nhóm đó mới chấm điểm intent / từ khóa.
    """

    accessible_tabs = ALL_TABS if role == "Sale" else TABS_PUBLIC

    candidates: List[Tuple[float, str, Dict[str, str]]] = []

    normalized_device = normalize_text(device or "")

    # --------------------------------------------------------------------------
    # Tầng 1: tìm row đúng thiết bị
    # --------------------------------------------------------------------------
    device_rows: List[Tuple[str, Dict[str, str]]] = []

    if normalized_device:
        for tab in accessible_tabs:
            for row in RAM_CACHE.get(tab, []):
                row_device = normalize_text(get_device_name(row))

                if not row_device:
                    continue

                if (
                    row_device == normalized_device
                    or normalized_device in row_device
                    or row_device in normalized_device
                ):
                    device_rows.append((tab, row))

    # Nếu có dữ liệu đúng thiết bị, chỉ tìm trong nhóm này.
    if device_rows:
        for tab, row in device_rows:
            score = score_row(
                query=query,
                row=row,
                device=device,
                intent=intent,
            )

            # Row đúng thiết bị luôn có điểm nền.
            score += 100

            candidates.append((score, tab, row))

    # --------------------------------------------------------------------------
    # Tầng 2: fallback khi chưa nhận diện được thiết bị
    # --------------------------------------------------------------------------
    else:
        for tab in accessible_tabs:
            for row in RAM_CACHE.get(tab, []):
                score = score_row(
                    query=query,
                    row=row,
                    device=device,
                    intent=intent,
                )

                if score > 0:
                    candidates.append((score, tab, row))

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    # Loại duplicate theo nội dung.
    selected: List[Tuple[float, str, Dict[str, str]]] = []
    seen = set()

    for item in candidates:
        score, tab, row = item

        fingerprint = (
            tab,
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
            ),
        )

        if fingerprint in seen:
            continue

        seen.add(fingerprint)
        selected.append(item)

        if len(selected) >= max_rows:
            break

    if not selected:
        return ""

    blocks: List[str] = []

    for score, tab, row in selected:
        block = [
            f"=== TAB: {tab} | RELEVANCE: {round(score, 1)} ==="
        ]

        for key, value in row.items():
            value = str(value).strip()

            if value:
                block.append(f"{key}: {value}")

        blocks.append("\n".join(block))

    return "\n\n".join(blocks)


# ==============================================================================
# URL CONTROL
# ==============================================================================

URL_PATTERN = re.compile(
    r"https?://[^\s<>\]\)\"']+",
    flags=re.IGNORECASE,
)


def extract_urls(text: str) -> List[str]:
    if not text:
        return []

    urls = URL_PATTERN.findall(text)

    # Loại dấu câu cuối URL.
    cleaned = []

    for url in urls:
        url = url.rstrip(".,;:!?")

        if url not in cleaned:
            cleaned.append(url)

    return cleaned


def sanitize_urls_against_knowledge(
    answer: str,
    knowledge_context: str,
) -> str:
    """
    Chỉ cho phép URL xuất hiện trong Knowledge Context.
    URL AI tự bịa sẽ bị xóa.
    """
    allowed = set(extract_urls(knowledge_context))

    def replace_url(match):
        url = match.group(0).rstrip(".,;:!?")

        if url in allowed:
            return url

        return "[link chưa có trong kho dữ liệu]"

    return URL_PATTERN.sub(replace_url, answer)


# ==============================================================================
# RESPONSE SANITIZATION
# ==============================================================================

FORBIDDEN_LINES = [
    "here's a thinking process:",
    "here is a thinking process:",
    "chain of thought",
    "control panel",
    "add a local printer",
    "devices and printers",
    "use an existing port",
    "add printer or scanner",
    "thêm máy in thủ công",
]


def sanitize_response_content(
    text: str,
    knowledge_context: str = "",
) -> str:
    if not text:
        return ""

    text = str(text)

    # Xóa think tags.
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    text = re.sub(
        r"<analysis>.*?</analysis>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Một số model có thể trả phần reasoning bằng tiếng Anh.
    if "Here's a thinking process:" in text:
        text = text.split(
            "Here's a thinking process:",
            1,
        )[0]

    lines = []

    for line in text.splitlines():
        lower = line.lower().strip()

        # Không xóa mọi dòng chứa "control panel" một cách mù quáng.
        # Chỉ xóa các câu hướng dẫn cấm rõ ràng.
        if any(
            phrase in lower
            for phrase in FORBIDDEN_LINES
        ):
            continue

        lines.append(line.rstrip())

    text = "\n".join(lines)

    # Không dùng heading Markdown kiểu #.
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*---+\s*$", "", text, flags=re.MULTILINE)

    # Chuẩn hóa xuống dòng.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Chống URL bịa.
    if knowledge_context:
        text = sanitize_urls_against_knowledge(
            text,
            knowledge_context,
        )

    return text.strip()


# ==============================================================================
# KNOWLEDGE QUALITY CHECK
# ==============================================================================

def knowledge_has_device(
    knowledge_context: str,
    device: Optional[str],
) -> bool:
    if not device:
        return bool(knowledge_context.strip())

    return normalize_text(device) in normalize_text(
        knowledge_context
    )


def build_no_data_response(
    device: Optional[str],
    intent: Optional[str],
) -> str:
    if device and intent == "driver":
        return (
            f"Dạ em đã nhận thiết bị **{device}**, "
            "nhưng hiện trong Kho dữ liệu chưa có đủ thông tin Driver "
            "để em gửi chính xác.\n\n"
            "Anh/chị vui lòng kiểm tra lại model hoặc gửi ảnh tem/model "
            "máy in, em sẽ khoanh đúng thiết bị trước ạ."
        )

    if device:
        return (
            f"Dạ em đã nhận thiết bị **{device}**, "
            "nhưng Kho dữ liệu hiện chưa có thông tin phù hợp với yêu cầu này.\n\n"
            "Anh/chị mô tả thêm lỗi hoặc nhu cầu cụ thể giúp em nhé ạ."
        )

    return (
        "Dạ hiện em chưa xác định được đúng thiết bị hoặc dữ liệu phù hợp.\n\n"
        "Anh/chị cho em xin **tên model máy** "
        "(ví dụ: SPR02, K200L...) và mô tả nhu cầu/lỗi giúp em nhé ạ."
    )


# ==============================================================================
# SYSTEM PROMPT
# ==============================================================================

def build_smart_system_prompt(
    knowledge_context: str,
    device: Optional[str],
    intent: Optional[str],
) -> str:
    intent_name = {
        "driver": "Cài Driver trên máy tính",
        "mobile": "Cài trên điện thoại / POS / LAN / đổi IP",
        "troubleshoot": "Khắc phục sự cố",
        "info": "Tra cứu thông tin / thông số",
        None: "Chưa xác định",
    }.get(intent, "Chưa xác định")

    return f"""
Bạn là Trợ Lý KHO Sapo, chuyên hỗ trợ thiết bị phần cứng
như máy in đơn hàng, máy in tem, máy quét mã vạch và thiết bị POS.

QUY TẮC QUAN TRỌNG NHẤT:

1. Chỉ sử dụng thông tin có trong KHO DỮ LIỆU được cung cấp.
2. Không tự bịa:
   - model
   - driver
   - link
   - thông số
   - tên phần mềm
   - quy trình kỹ thuật
3. Nếu KHO DỮ LIỆU không có thông tin thì nói rõ là chưa có dữ liệu.
4. Không được biến suy đoán thành sự thật.
5. Nếu có nhiều thiết bị gần giống nhau, không tự chọn bừa.
6. Không cung cấp URL nếu URL đó không xuất hiện trong KHO DỮ LIỆU.
7. Không xuất suy nghĩ nội bộ hoặc chain-of-thought.
8. Trả lời bằng tiếng Việt.
9. Xưng "Em", gọi người dùng là "Anh/chị".
10. Trả lời thực tế, ngắn gọn, dễ làm theo.
11. Không dùng bảng Markdown.
12. Không dùng tiêu đề Markdown bằng #.
13. Không hướng dẫn Control Panel / Add a local printer / Devices and Printers,
    trừ khi chính KHO DỮ LIỆU có một quy trình bắt buộc như vậy.
14. Nếu người dùng đã có model trong ngữ cảnh thì không hỏi lại model một lần nữa.
15. Nếu câu hỏi là cài Driver, ưu tiên đưa đúng Driver có trong kho
    và nền tảng Windows/macOS nếu dữ liệu có.
16. Nếu câu hỏi là lỗi, tập trung đúng lỗi người dùng mô tả,
    không xả toàn bộ danh sách lỗi của thiết bị.

THIẾT BỊ HIỆN TẠI:
{device or "Chưa xác định"}

Ý ĐỊNH:
{intent_name}

KHO DỮ LIỆU SAPO:
{knowledge_context or "(Không tìm thấy dữ liệu phù hợp)"}
""".strip()


# ==============================================================================
# LLM CALLS
# ==============================================================================

async def call_gemini(
    system_instruction: str,
    user_message: str,
) -> str:
    if not GEMINI_API_KEY or not HTTP_CLIENT:
        return ""

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "systemInstruction": {
            "parts": [
                {
                    "text": system_instruction,
                }
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": user_message,
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 1800,
        },
    }

    try:
        res = await HTTP_CLIENT.post(
            url,
            headers=headers,
            json=payload,
            timeout=15.0,
        )

        if res.status_code != 200:
            return ""

        data = res.json()

        candidates = data.get("candidates", [])

        if not candidates:
            return ""

        parts = (
            candidates[0]
            .get("content", {})
            .get("parts", [])
        )

        texts = [
            str(part.get("text", ""))
            for part in parts
            if part.get("text")
        ]

        return "\n".join(texts).strip()

    except Exception:
        return ""


async def call_cerebras(
    system_instruction: str,
    user_message: str,
) -> str:
    if not CEREBRAS_API_KEY or not CEREBRAS_MODEL or not HTTP_CLIENT:
        return ""

    url = "https://api.cerebras.ai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {CEREBRAS_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": CEREBRAS_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_instruction,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        "temperature": 0.1,
        "max_tokens": 1800,
        "stream": False,
    }

    try:
        res = await HTTP_CLIENT.post(
            url,
            headers=headers,
            json=payload,
            timeout=12.0,
        )

        if res.status_code != 200:
            return ""

        data = res.json()

        choices = data.get("choices", [])

        if not choices:
            return ""

        return str(
            choices[0]
            .get("message", {})
            .get("content", "")
        ).strip()

    except Exception:
        return ""


async def call_llm_single(
    system_instruction: str,
    user_message: str,
    knowledge_context: str = "",
) -> str:
    # Cerebras trước.
    answer = await call_cerebras(
        system_instruction,
        user_message,
    )

    if not answer:
        # Gemini fallback.
        answer = await call_gemini(
            system_instruction,
            user_message,
        )

    if not answer:
        return ""

    return sanitize_response_content(
        answer,
        knowledge_context,
    )


# ==============================================================================
# DETERMINISTIC ANSWER FOR MENU / MISSING DEVICE
# ==============================================================================

def make_menu_response(device: str) -> str:
    return MENU_TEXT.format(device=device)


# ==============================================================================
# CHAT HISTORY HELPERS
# ==============================================================================

def extract_web_chat_latest_message(
    messages: list,
) -> str:
    if not messages:
        return ""

    for message in reversed(messages):
        if not isinstance(message, dict):
            continue

        text = message.get("text")

        if text:
            return str(text)

    return ""


def get_web_history(messages: list) -> List[Dict[str, str]]:
    result = []

    for message in messages[-8:]:
        if not isinstance(message, dict):
            continue

        text = str(message.get("text", "")).strip()

        if not text:
            continue

        role = message.get("role", "user")

        if role not in {"user", "assistant"}:
            role = "user"

        result.append({
            "role": role,
            "text": text,
        })

    return result


# ==============================================================================
# RESPONSE WRAPPER FOR GOOGLE CHAT
# ==============================================================================

def wrap_gsuite_addon_response(text_message: str) -> dict:
    clean_text = sanitize_response_content(text_message)

    # Google Chat không cần Markdown link [text](url) theo format này.
    clean_text = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        r"\1: \2",
        clean_text,
    )

    return {
        "hostAppDataAction": {
            "chatDataAction": {
                "createMessageAction": {
                    "message": {
                        "text": clean_text,
                    }
                }
            }
        }
    }


# ==============================================================================
# CORE ENGINE
# ==============================================================================

async def process_message(
    message: str,
    role: str = "Khach_Hang",
    session: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Đây là trung tâm xử lý chung cho Web Chat và Google Chat.

    Trình tự:
    1. greeting
    2. nhận diện thiết bị
    3. lấy device từ session nếu user đang trả lời menu
    4. nhận diện menu 1/2/3
    5. nhận diện intent
    6. retrieval đúng thiết bị
    7. LLM chỉ diễn đạt dựa trên dữ liệu
    """

    message = clean_google_chat_text(message)
    session = session or {
        "device": None,
        "intent": None,
        "history": [],
    }

    # --------------------------------------------------------------------------
    # Greeting
    # --------------------------------------------------------------------------
    if is_greeting(message):
        answer = (
            "👋 Dạ em chào anh/chị! Em là **Trợ Lý KHO Sapo**.\n\n"
            "Anh/chị gửi tên model thiết bị hoặc mô tả lỗi cần hỗ trợ, "
            "em sẽ tra cứu và hướng dẫn ngay ạ."
        )

        return answer, session

    # --------------------------------------------------------------------------
    # Device detection
    # --------------------------------------------------------------------------
    found_device = find_device_in_query(
        message,
        role=role,
    )

    if found_device:
        session["device"] = found_device

    cached_device = session.get("device")

    # --------------------------------------------------------------------------
    # Menu 1/2/3 - xử lý bằng Python, KHÔNG giao cho AI.
    # --------------------------------------------------------------------------
    choice = normalize_menu_choice(message)

    if choice:
        if not cached_device:
            answer = (
                "Dạ em chưa biết anh/chị đang chọn cho thiết bị nào ạ.\n\n"
                "Anh/chị gửi giúp em tên model máy trước "
                "(ví dụ: SPR02, K200L...) nhé."
            )

            return answer, session

        intent = choice
        session["intent"] = intent

        final_query = build_intent_query(
            device=cached_device,
            intent=intent,
            original_message=message,
        )

    else:
        # ----------------------------------------------------------------------
        # Intent từ câu tự nhiên.
        # ----------------------------------------------------------------------
        detected_intent = detect_intent(message)

        if detected_intent:
            session["intent"] = detected_intent

        intent = detected_intent or session.get("intent")

        # ----------------------------------------------------------------------
        # Chỉ có tên thiết bị -> menu.
        # ----------------------------------------------------------------------
        normalized_message = normalize_text(message)
        normalized_device = normalize_text(cached_device or "")

        is_only_device = (
            bool(cached_device)
            and bool(normalized_device)
            and normalized_message == normalized_device
        )

        if is_only_device:
            session["intent"] = None
            return make_menu_response(cached_device), session

        # ----------------------------------------------------------------------
        # Không có device và câu hỏi không đủ để xác định -> hỏi model.
        # ----------------------------------------------------------------------
        if not cached_device:
            # Nếu user đang hỏi rất chung chung, không gọi AI bừa.
            return (
                "Dạ anh/chị cho em xin **tên model thiết bị** "
                "(ví dụ: SPR02, K200L...) để em tra đúng dữ liệu nhé ạ.",
                session,
            )

        final_query = build_intent_query(
            device=cached_device,
            intent=intent,
            original_message=message,
        )

    # --------------------------------------------------------------------------
    # Retrieval.
    # --------------------------------------------------------------------------
    knowledge = get_high_precision_knowledge(
        query=final_query,
        role=role,
        device=cached_device,
        intent=session.get("intent"),
        max_rows=8,
    )

    # --------------------------------------------------------------------------
    # Nếu không có dữ liệu -> không gọi LLM để tránh hallucination.
    # --------------------------------------------------------------------------
    if not knowledge.strip():
        answer = build_no_data_response(
            device=cached_device,
            intent=session.get("intent"),
        )

        return answer, session

    # --------------------------------------------------------------------------
    # Prompt + LLM.
    # --------------------------------------------------------------------------
    system_instruction = build_smart_system_prompt(
        knowledge_context=knowledge,
        device=cached_device,
        intent=session.get("intent"),
    )

    # Cho LLM biết context hội thoại gần nhất nhưng không để nó tự đổi device.
    history_text = ""

    history = session.get("history", [])

    if history:
        history_lines = []

        for item in history[-6:]:
            role_name = (
                "Khách"
                if item.get("role") == "user"
                else "Trợ lý"
            )

            history_lines.append(
                f"{role_name}: {item.get('text', '')}"
            )

        history_text = (
            "\n\nLỊCH SỬ GẦN NHẤT:\n"
            + "\n".join(history_lines)
        )

    user_prompt = (
        f"Thiết bị đã xác định: {cached_device}\n"
        f"Ý định đã xác định: {session.get('intent') or 'chưa xác định'}\n"
        f"Câu hỏi hiện tại: {message}\n"
        f"{history_text}\n\n"
        "Hãy trả lời trực tiếp câu hỏi hiện tại dựa trên KHO DỮ LIỆU."
    )

    answer = await call_llm_single(
        system_instruction=system_instruction,
        user_message=user_prompt,
        knowledge_context=knowledge,
    )

    # --------------------------------------------------------------------------
    # LLM lỗi -> trả câu an toàn, không bịa.
    # --------------------------------------------------------------------------
    if not answer:
        answer = (
            f"Dạ em đã xác định thiết bị **{cached_device}** nhưng "
            "hiện hệ thống AI chưa trả được câu trả lời.\n\n"
            "Anh/chị thử gửi lại yêu cầu sau ít phút giúp em nhé ạ."
        )

    return answer, session


# ==============================================================================
# WEB CHAT
# ==============================================================================

@app.post("/chat")
async def chat_stream(req: ChatRequest):
    latest_msg = extract_web_chat_latest_message(req.messages)

    if not latest_msg:
        async def empty_gen():
            yield (
                "Dạ anh/chị gửi giúp em tên thiết bị hoặc câu hỏi "
                "cần hỗ trợ nhé ạ."
            )

        return StreamingResponse(
            empty_gen(),
            media_type="text/plain",
        )

    # Web Chat có history riêng theo request.
    # Tạo session tạm để hiểu menu trong chính conversation.
    temp_session = {
        "device": None,
        "intent": None,
        "updated_at": time.time(),
        "history": get_web_history(req.messages[:-1]),
    }

    # Tìm device trong toàn bộ history trước.
    historical_text = " ".join(
        item.get("text", "")
        for item in temp_session["history"]
        if item.get("role") == "user"
    )

    historical_device = find_device_in_query(
        historical_text,
        role=req.role,
    )

    if historical_device:
        temp_session["device"] = historical_device

    answer, _ = await process_message(
        message=latest_msg,
        role=req.role,
        session=temp_session,
    )

    async def response_gen():
        yield answer

    return StreamingResponse(
        response_gen(),
        media_type="text/plain",
    )


# ==============================================================================
# GOOGLE CHAT BOT
# ==============================================================================

@app.post("/google-chat")
async def google_chat_webhook(request: Request):
    try:
        event = await request.json()

        user_message = extract_user_text(event)
        cleaned_message = clean_google_chat_text(user_message)

        # space.name là key ưu tiên.
        space_id = (
            event.get("space", {}).get("name")
            or event.get("user", {}).get("name")
            or "default_space"
        )

        event_type = (
            event.get("type")
            or event.get("chat", {}).get("type")
            or ""
        )

        # ----------------------------------------------------------------------
        # Bot vừa được add vào Space.
        # ----------------------------------------------------------------------
        if event_type == "ADDED_TO_SPACE":
            answer = (
                "👋 Xin chào! Em là **Trợ Lý KHO Sapo**.\n\n"
                "Anh/chị hãy gửi tên thiết bị hoặc mô tả lỗi, "
                "em sẽ tra cứu và hỗ trợ ngay ạ."
            )

            return JSONResponse(
                content=wrap_gsuite_addon_response(answer)
            )

        # ----------------------------------------------------------------------
        # Lấy session.
        # ----------------------------------------------------------------------
        session = get_session(space_id)

        # ----------------------------------------------------------------------
        # Process.
        # ----------------------------------------------------------------------
        answer, updated_session = await process_message(
            message=cleaned_message,
            role="Sale",
            session=session,
        )

        # ----------------------------------------------------------------------
        # Lưu lịch sử sau khi trả lời.
        # ----------------------------------------------------------------------
        update_session(
            space_id=space_id,
            device=updated_session.get("device"),
            intent=updated_session.get("intent"),
            user_message=cleaned_message,
            assistant_message=answer,
        )

        return JSONResponse(
            content=wrap_gsuite_addon_response(answer)
        )

    except Exception:
        # Không để exception nội bộ lộ ra cho người dùng.
        return JSONResponse(
            content=wrap_gsuite_addon_response(
                "Dạ hệ thống đang gặp lỗi khi xử lý yêu cầu.\n\n"
                "Anh/chị thử gửi lại tin nhắn sau ít phút giúp em nhé ạ."
            )
        )


# ==============================================================================
# OPTIONAL DEBUG ENDPOINT
# Chỉ nên bật khi test. Có thể xóa nếu deploy production.
# ==============================================================================

@app.get("/debug/cache")
def debug_cache():
    return {
        "version": APP_VERSION,
        "tabs": {
            tab: len(rows)
            for tab, rows in RAM_CACHE.items()
        },
        "sessions": {
            key: {
                "device": value.get("device"),
                "intent": value.get("intent"),
                "updated_at": value.get("updated_at"),
            }
            for key, value in GOOGLE_CHAT_SESSION_CACHE.items()
        },
    }
