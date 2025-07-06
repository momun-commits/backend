from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI()

class ComplaintRequest(BaseModel):
    text: str

@app.post("/summarize")
async def summarize(complaint: ComplaintRequest):
    url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": complaint.text}

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    
    try:
        summary = response.json()[0]['summary_text']
    except Exception:
        raise HTTPException(status_code=500, detail="Unexpected response: " + str(response.json()))
    
    return {"summary": summary}

@app.post("/respond")
async def respond(complaint: ComplaintRequest):
    reply = f"Thank you for your feedback: '{complaint.text}'. We’ll get back to you soon."
    return {"response": reply}
