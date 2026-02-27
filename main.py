import os
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from google import genai

load_dotenv()

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow grader
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class AskRequest(BaseModel):
    video_url: str
    topic: str


def extract_video_id(url: str):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None


def seconds_to_hhmmss(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@app.post("/ask")
async def ask(request: AskRequest):

    try:
        video_id = extract_video_id(request.video_url)
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")

        # 1️⃣ Get transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # 2️⃣ Combine transcript text
        full_text = ""
        for entry in transcript:
            full_text += f"[{entry['start']}] {entry['text']} "

        # 3️⃣ Ask Gemini to find timestamp in transcript
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"""
            Given the transcript below, find the FIRST time the topic "{request.topic}" appears.
            Return ONLY the timestamp in seconds (number only).

            Transcript:
            {full_text[:20000]}
            """
        )

        seconds_match = re.search(r"\d+", response.text)
        if not seconds_match:
            raise HTTPException(status_code=500, detail="Timestamp not found")

        seconds = int(seconds_match.group())
        timestamp = seconds_to_hhmmss(seconds)

        return {
            "timestamp": timestamp,
            "video_url": request.video_url,
            "topic": request.topic
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))