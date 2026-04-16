from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel
import base64
from io import BytesIO
import numpy as np

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

class AudioChunk(BaseModel):
    data: str  # base64 audio

# Your existing models (from Colab)
# asr, embedder, zeroshot = load_models()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            audio_b64 = data['audio']
            
            # Decode audio
            audio_bytes = base64.b64decode(audio_b64)
            
            # Transcribe (30ms chunk)
            result = asr(audio_bytes)
            text = result['text']
            
            # Score instantly
            relevance = score_relevance(text)
            labels = zeroshot(text, gd_labels)
            
            # Send back to frontend
            await websocket.send_json({
                "transcript": text,
                "relevance": relevance,
                "labels": labels,
                "speaker": "Student"
            })
            
            # AI Moderator responds
            ai_response = generate_ai_response(text)
            await websocket.send_json({
                "transcript": ai_response,
                "speaker": "AI Moderator",
                "is_ai": True
            })
            
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)