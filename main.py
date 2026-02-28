import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = FastAPI()

# CORS (required for grader)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    try:
        # Ask Gemini directly using YouTube URL
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"""
            Watch this YouTube video: {request.video_url}

            Find the FIRST time the topic "{request.topic}" is spoken.

            Return ONLY the timestamp in HH:MM:SS format.
            Example: 00:05:47
            Do not return anything else.
            """
        )

        # Extract HH:MM:SS safely
        match = re.search(r"\d{2}:\d{2}:\d{2}", response.text)
        if not match:
            raise HTTPException(status_code=500, detail="Timestamp not found")

        timestamp = match.group()

        return {
            "timestamp": timestamp,
            "video_url": request.video_url,
            "topic": request.topic
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))