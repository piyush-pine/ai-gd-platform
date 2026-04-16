from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
import numpy as np
import uvicorn
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import asyncio

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Global models (load once)
device = 0 if torch.cuda.is_available() else -1
asr = None
embedder = None
zeroshot = None
gd_labels = ["leadership", "teamwork", "clear communication", "vague", "strong argument", "off-topic"]
ideal_points = ["AI improves scalability", "humans judge soft skills", "hybrid approach works best"]

class AudioChunk(BaseModel):
    data: str  # base64 audio

@app.on_event("startup")
async def load_models():
    global asr, embedder, zeroshot
    print("🔄 Loading AI models...")
    
    asr = pipeline("automatic-speech-recognition", 
                   model="openai/whisper-small",
                   chunk_length_s=5,
                   device=device)
    
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    zeroshot = pipeline("zero-shot-classification", 
                       model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
                       device=device)
    
    print("✅ All models loaded!")

@app.post("/transcribe")
async def transcribe_audio(chunk: AudioChunk):
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(chunk.data)
        
        # Transcribe
        result = asr(audio_bytes)
        text = result['text'].strip()
        
        if len(text.split()) < 3:
            return {"transcript": "", "scores": {"relevance": 0, "clarity": 0, "leadership": 0}}
        
        # Score relevance
        text_emb = embedder.encode(text)
        ideal_embs = embedder.encode(ideal_points)
        relevance = float(util.cos_sim(text_emb, ideal_embs).max().cpu())
        
        # Zero-shot labels
        z_result = zeroshot(text, gd_labels[:5], multi_label=True)
        clarity = max([s for l, s in zip(z_result['labels'], z_result['scores']) if 'clear' in l.lower()] or [0])
        leadership = max([s for l, s in zip(z_result['labels'], z_result['scores']) if 'leadership' in l.lower()] or [0])
        
        return {
            "transcript": text,
            "scores": {
                "relevance": round(relevance * 100, 1),
                "clarity": round(clarity * 100, 1),
                "leadership": round(leadership * 100, 1),
                "overall": round((relevance + clarity + leadership) / 3 * 100, 1)
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if data.get('type') == 'audio':
                result = await transcribe_audio(AudioChunk(data=data))
                await websocket.send_json(result)
                
                # AI Moderator response
                ai_response = generate_ai_response(result.get('transcript', ''))
                await websocket.send_json({
                    "transcript": ai_response,
                    "speaker": "AI Moderator",
                    "is_ai": True
                })
    except WebSocketDisconnect:
        pass

def generate_ai_response(text):
    responses = {
        "good": "Excellent point! Can someone counter this?",
        "avg": "Interesting. Can you give a concrete example?",
        "weak": "Let's connect this back to the main topic."
    }
    return responses["avg"]  # Simple for now

@app.get("/")
async def root():
    return {"message": "AI GD Platform Live!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
