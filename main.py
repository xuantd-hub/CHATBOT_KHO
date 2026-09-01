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

app = FastAPI(title="Trợ Lý KHO Sapo Universal Engine", version="900.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# CẤU HÌNH BIẾN MÔI TRƯỜNG & LƯU BỘ NHỚ RAM
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
        "version": "900.0", 
        "engine": "Cerebras Dedicated Universal Engine",
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
# LÀM SẠCH VĂN BẢN VÀ TRÍCH XUẤT CÂU HỎI
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
# THUẬT TOÁN LỌC DỮ LIỆU TỔNG QUÁT THEO DANH MỤC (RAG MULTI-TAB)
# ------------------------------------------------------------------------------
def get_high_precision_knowledge(query: str, role: str) -> str:
    accessible_tabs = ALL_TABS if role == "Sale" else TABS_PUBLIC
    query_lower = query.lower()
    
    stop_words = {"mình", "có", "bị", "được", "không", "cho", "với", "là", "và", "nhé", "ạ", "cần", "giúp", "tôi", "xin", "lỗi", "thế", "nào", "bao", "nhiêu", "thông", "số", "qua", "đã", "ok"}
    words = [w for w in query_lower.split() if len(w) > 1 and w not in stop_words]
    if not words: words = [query_lower]

    scored_rows = []
    for tab in accessible_tabs:
        for row in RAM_CACHE.get(tab, []):
            row_text = " ".join(str(v).lower() for v in row.values())
            score = 0
            
            # Tính điểm khớp từ khóa trong toàn bộ dòng
            for w in words:
                if w in row_text:
                    score += 10
            
            # Ưu tiên cộng điểm cao nếu khớp Tên Thiết Bị / Loại Thao Tác / Tên Lỗi / Tên Chính Sách
            key_field = str(row.get("Ten_Thiet_Bi", row.get("Loai_Thiet_Bi", row.get("Ten_Loi", row.get("Ten_Chinh_Sach", row.get("Tu_Khoa_Nhan_Dien", "")))))).lower()
            for w in words:
                if len(w) >= 2 and w in key_field:
                    score += 100

            if score > 0:
                scored_rows.append((score, tab, row))

    scored_rows.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored_rows[:3]

    knowledge_text = ""
    for score, tab, row in top_matches:
        knowledge_text += f"\n=== DỮ LIỆU THUỘC TAB [{tab}] ===\n"
        for key, value in row.items():
            if value: 
                knowledge_text += f"- {key}: {value}\n"

    return knowledge_text

# ------------------------------------------------------------------------------
# PROMPT HỆ THỐNG TỔNG QUÁT BẢO VỆ TOÀN BỘ THƯ VIỆN
# ------------------------------------------------------------------------------
def build_smart_system_prompt(knowledge_context: str) -> str:
    return f"""
Bạn là **Trợ Lý KHO Sapo** – Chuyên gia IT cao cấp hỗ trợ Kỹ thuật, Lỗi thiết bị, Cài đặt và Chính sách của Sapo.
Xưng hô: Xưng "Em", gọi "Anh/chị". Phong cách: Lịch sự, ngắn gọn, chính xác, tinh tế.

🎯 BỘ QUY TẮC PHÂN LOẠI Ý ĐỊNH VÀ PHẢN HỒI (ÁP DỤNG TOÀN HỆ THỐNG):

1. TRƯỜNG HỢP 1: CÂU HỎI MẬP MỜ / CHỈ NÓI TÊN THIẾT BỊ / CHỈ NÓI TÊN CHÍNH SÁCH
(Ví dụ: "spr02", "k200l", "chính sách đổi trả", "máy in xprinter", "bảo hành"):
👉 TUYỆT ĐỐI KHÔNG tự đoán mò nhu cầu! KHÔNG xả ngay bài hướng dẫn dài dòng hay tự ý gán kịch bản cài mobile/PC!
👉 BẮT BUỘC hỏi lại 1 câu khoanh vùng nhu cầu tùy theo đối tượng:
   - Nếu là THIẾT BỊ (Ví dụ: SPR02, K200L...):
     "Dạ, với thiết bị **[Tên thiết bị]**, anh/chị cần em hỗ trợ mục nào dưới đây ạ?
      1. 💻 Cài đặt trên Máy tính (Windows / Mac)
      2. 📱 Cài đặt trên Điện thoại / App (App XTEST / Kết nối LAN)
      3. 🛠️ Sửa lỗi kỹ thuật / Tra cứu thông số"
   - Nếu là CHÍNH SÁCH / NỘI BỘ / LỖI CHUNG (Bảo hành, Thu hồi, Chiết khấu...):
     "Dạ, về **[Chủ đề]**, anh/chị đang cần tra cứu quy định hoặc hướng dẫn cụ thể nào ạ?"

2. TRƯỜNG HỢP 2: CÂU HỎI ĐÃ CÓ Ý ĐỊNH RÕ RÀNG
(Ví dụ: "cài driver spr02 máy tính", "spr02 in ra giấy trắng", "chính sách bảo hành máy in 12 tháng"):
👉 Trả lời TRỰC DIỆN, tóm tắt 3-4 bước ngắn gọn, rõ ràng.
👉 ĐÍNH KÈM TÀI LIỆU VÀ MỆNH ĐỀ MATCH 100% (MATCHING STRICT RULE):
   - Đang hỏi CÀI TRÊN ĐIỆN THOẠI ➔ CHỈ đính kèm link/video App Mobile. TUYỆT ĐỐI CẤM gửi link Driver Win/Mac hoặc ảnh màn hình Windows Properties!
   - Đang hỏi CÀI TRÊN MÁY TÍNH ➔ CHỈ đính kèm link Driver Win/Mac & video thao tác PC.
   - Đang hỏi SỬA LỖI ➔ CHỈ gửi hướng dẫn xử lý lỗi đó, KHÔNG đính kèm link cài đặt ban đầu.
   - Đang hỏi CHÍNH SÁCH / NỘI BỘ ➔ Trích xuất đúng điều khoản trong Tab [3_CHINH_SACH_SAPO] hoặc [4_DU_LIEU_NOI_BO].

3. LUẬT THÉP BẢO VỆ DỮ LIỆU CHỐNG HALLUCINATION:
- KHÔNG TỰ BỊA BƯỚC THỦ CÔNG: Không tự sáng tác các bước Windows Control Panel hay thao tác phần cứng nếu trong Dữ liệu gốc không yêu cầu.
- KHÔNG TỰ BỊA LINK: Chỉ xuất các đường URL/Drive/Youtube thực sự xuất hiện trong KHO DỮ LIỆU bên dưới. Nếu dữ liệu không có link, chỉ trả lời chữ.
- KHÔNG LẪN LỘN MEDIA: Nếu không có hình ảnh/video khớp 100% với thao tác đang hỏi, thà KHÔNG GỬI ẢNH chứ tuyệt đối không vứt ảnh rác hoặc ảnh của thiết bị/HĐH khác vào.

---

KHO DỮ LIỆU TRÍCH XUẤT TỪ SHEET (CỦA TẤT CẢ CÁC TAB):
{knowledge_context}
"""

# ------------------------------------------------------------------------------
# HÀM GỌI CEREBRAS LLM (CÓ GEMINI DỰ PHÒNG)
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

    # 2. DỰ PHÒNG GEMINI NẾU CEREBRAS GẶP SỰ CỐ TẠM THỜI
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
