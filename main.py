import os
import time
import uuid
import re
import yt_dlp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for grading
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class AskRequest(BaseModel):
    video_url: str
    topic: str

@app.post("/ask")
async def ask(request: AskRequest):

    temp_filename = f"audio_{uuid.uuid4()}.mp3"

    try:
        # 1️⃣ Download audio only
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": temp_filename,
            "quiet": True,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([request.video_url])

        # 2️⃣ Upload file
        uploaded_file = client.files.upload(file=temp_filename)

        # 3️⃣ Wait until ACTIVE
        while uploaded_file.state != types.FileState.ACTIVE:
            time.sleep(3)
            uploaded_file = client.files.get(name=uploaded_file.name)

        # 4️⃣ Ask Gemini (no structured schema — enforce format via prompt)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                uploaded_file,
                f"""
                Find the FIRST time the topic "{request.topic}" is spoken.
                Return ONLY the timestamp in HH:MM:SS format.
                Example: 00:05:47
                """
            ],
        )

        text_output = response.text

        match = re.search(r"\d{2}:\d{2}:\d{2}", text_output)
        if not match:
            raise HTTPException(status_code=500, detail="Invalid timestamp format")

        timestamp = match.group()

        return {
            "timestamp": timestamp,
            "video_url": request.video_url,
            "topic": request.topic
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)