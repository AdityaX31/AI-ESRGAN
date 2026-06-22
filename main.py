import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class RequestBody(BaseModel):
    prompt: str


# -------------------------
# GEMINI CALL
# -------------------------
def call_gemini(prompt: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    res = requests.post(url, json=payload, timeout=15)
    res.raise_for_status()

    data = res.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


# -------------------------
# GROQ CALL (FALLBACK)
# -------------------------
def call_groq(prompt: str):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-70b-versatile",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    res = requests.post(url, json=payload, headers=headers, timeout=15)
    res.raise_for_status()

    data = res.json()
    return data["choices"][0]["message"]["content"]


# -------------------------
# MAIN ROUTE
# -------------------------
@app.post("/generate")
def generate(req: RequestBody):

    # 1. TRY GEMINI
    try:
        print("Using Gemini")
        result = call_gemini(req.prompt)
        return {
            "provider": "gemini",
            "result": result
        }

    except Exception as e:
        print("Gemini failed:", str(e))

    # 2. FALLBACK GROQ
    try:
        print("Fallback to Groq")
        result = call_groq(req.prompt)
        return {
            "provider": "groq",
            "result": result
        }

    except Exception as e:
        print("Groq failed:", str(e))

    # 3. ALL FAILED
    return {
        "provider": "none",
        "result": "AI service currently unavailable"
    }