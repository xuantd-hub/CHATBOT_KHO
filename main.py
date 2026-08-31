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

app = FastAPI(title="Trợ Lý KHO Sapo Minimal Engine", version="155.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SHEET_ID = os.getenv("SHEET_ID", "1ZMq0mTiQTDiP92UPaOIv39Q17WJXDiuvrcyYwfs7_Ag").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()

RAM_CACHE_SHEETS = {}

TABS_PUBLIC = [
    "1_THIET_BI_VA_LOI", 
    "2_HUONG_DAN_CAI_DAT", 
    "3_CHINH_SACH_SAPO", 
    "NHAN_DIEN_THIET_BI"
]
TAB_PRIVATE = "4_DU_LIEU_NOI_BO"
ALL_TABS = TABS_PUBLIC + [TAB_PRIVATE]

SYNONYMS_DICT = {
    "kẹt dao": ["không cắt giấy", "lỗi cắt giấy", "kẹt dao", "hư dao cắt", "cutter"],
    "khổ giấy": ["kích thước giấy", "khổ tem", "khổ giấy in", "paper size", "kích thước tem"],
    "điện thoại": ["xtest", "app xtest", "in qua lan", "đổi ip", "android", "ios", "wifi", "không dây"],
    "máy tính": ["driver", "windows", "mac", "pc", "laptop", "cài driver", "cáp usb"],
    "in ra giấy trắng": ["không ra mực", "trắng tinh", "mờ mực", "ngược giấy"],
    "cài đặt": ["cài máy", "setup", "hướng dẫn cài", "cách cài", "cấu hình", "kết nối"]
}

async def background_load_sheets():
    global RAM_CACHE_SHEETS
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            for tab in ALL_TABS:
                url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={tab}"
                res = await client.get(url)
                if res.status_code == 200:
                    df = pd.read_csv(io.BytesIO(res.content)).fillna("")
                    records = [{str(k): str(v).strip() for k, v in row.items() if str(v).strip()} for _, row in df.iterrows() if any(str(v).strip() for v in row.values)]
                    RAM_CACHE_SHEETS[tab] = records
    except Exception:
        pass

@app.on_event("startup")
async def startup_event():
    # Chạy ngầm việc tải dữ liệu để Uvicorn mở cổng 8080 ngay lập tức, chống lỗi timeout Cloud Run
    asyncio.create_task(background_load_sheets())

@app.get("/")
def health_check():
    return {"status": "healthy", "model": GEMINI_MODEL}

class ChatRequest(BaseModel):
    messages: list
    role: str = "Khach_Hang"

def extract_user_text(event: dict) -> str:
    if "message" in event and isinstance(event["message"], dict):
        if "text" in event["message"]: return event["message"]["text"]
        if "argumentText" in event["message"]: return event["message"]["argumentText"]
    if "chat" in event and isinstance(event["chat"], dict):
        chat = event["chat"]
        if "messagePayload" in chat and isinstance(chat["messagePayload"], dict):
            if "message" in chat["messagePayload"] and isinstance(chat["messagePayload"]["message"], dict):
                if "text" in chat["messagePayload"]["message"]: return chat["messagePayload"]["message"]["text"]
        if "message" in chat and isinstance(chat["message"], dict):
            if "text" in chat["message"]: return chat["message"]["text"]
    return "Hỗ trợ thiết bị"

def get_knowledge(query: str) -> str:
    query_lower = query.lower()
    text_out = ""
    for tab, rows in RAM_CACHE_SHEETS.items():
        for row in rows:
            row_str = " ".join(str(v) for v in row.values()).lower()
            if any(kw in row_str for kw in query_lower.split() if len(kw) > 2):
                for k, v in row.items():
                    if v: text_out += f"- {k}: {v}\n"
                break
        if text_out: break
    return text_out

async def call_gemini(system_prompt: str, user_msg: str) -> str:
    if not GEMINI_API_KEY:
        return "👋 Dạ em là Trợ Lý KHO Sapo. Anh/chị cần hỗ trợ tra cứu thiết bị nào ạ?"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 800}
    }
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            res = await client.post(url, headers=headers, json=payload)
            if res.status_code == 200:
                data = res.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        pass
    return "Dạ hệ thống đang tra cứu, anh/chị vui lòng thử lại nhé!"

@app.post("/chat")
async def chat_stream(req: ChatRequest):
    latest_msg = req.messages[-1]["text"] if req.messages else ""
    knowledge = get_knowledge(latest_msg)
    prompt = f"Bạn là Trợ Lý KHO Sapo. Trả lời ngắn gọn, thân thiện, không bịa đặt link.\n\nDữ liệu kho:\n{knowledge}"
    
    async def response_generator():
        ans = await call_gemini(prompt, latest_msg)
        yield ans
        
    return StreamingResponse(response_generator(), media_type="text/plain")

@app.post("/google-chat")
async def google_chat_webhook(request: Request):
    try:
        event = await request.json()
        text = extract_user_text(event)
        knowledge = get_knowledge(text)
        prompt = f"Bạn là Trợ Lý KHO Sapo. Trả lời ngắn gọn, thân thiện, không bịa đặt link.\n\nDữ liệu kho:\n{knowledge}"
        ans = await call_gemini(prompt, text)
        return JSONResponse(content={"hostAppDataAction": {"chatDataAction": {"createMessageAction": {"message": {"text": ans}}}}})
    except Exception:
        return JSONResponse(content={"hostAppDataAction": {"chatDataAction": {"createMessageAction": {"message": {"text": "Dạ hệ thống đang bận, anh/chị thử lại nhé!"}}}}})
