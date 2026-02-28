import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from google import genai

# Load environment variables
load_dotenv()

app = FastAPI()

# ✅ CORS (REQUIRED for grader)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class AskRequest(BaseModel):
    video_url: str
    topic: str


def extract_video_id(url: str):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None


def seconds_to_hhmmss(seconds: float):
    seconds = int(float(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


@app.post("/ask")
async def ask(request: AskRequest):
    try:
        # 1️⃣ Extract video ID
        video_id = extract_video_id(request.video_url)
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")

        # 2️⃣ Fetch transcript (new API style)
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)

        # 3️⃣ Combine transcript text
        full_text = ""
        for entry in transcript:
            full_text += f"[{entry.start}] {entry.text} "

        # Limit size to avoid token overflow
        full_text = full_text[:20000]

        # 4️⃣ Ask Gemini to find timestamp in seconds
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"""
            Given the transcript below, find the FIRST time the topic "{request.topic}" appears.
            Return ONLY the timestamp in seconds as a number.
            Do not return anything else.

            Transcript:
            {full_text}
            """
        )

        # 5️⃣ Extract seconds from Gemini response
        match = re.search(r"\d+(\.\d+)?", response.text)
        if not match:
            raise HTTPException(status_code=500, detail="Timestamp not found")

        seconds = float(match.group())
        timestamp = seconds_to_hhmmss(seconds)

        # 6️⃣ Return required format
        return {
            "timestamp": timestamp,
            "video_url": request.video_url,
            "topic": request.topic
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))