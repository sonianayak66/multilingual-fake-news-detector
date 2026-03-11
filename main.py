from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import newspaper
from detector import detect_fake_news

app = FastAPI(
    title="Multilingual Fake News Detector",
    description="Detects fake news in Indian regional languages",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

class UrlInput(BaseModel):
    url: str

@app.get("/")
def home():
    return {"status": "Fake News Detector is running!"}

@app.post("/analyze/text")
def analyze_text(data: TextInput):
    if len(data.text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Text too short.")
    result = detect_fake_news(data.text)
    return result

@app.post("/analyze/url")
def analyze_url(data: UrlInput):
    try:
        article = newspaper.Article(data.url)
        article.download()
        article.parse()
        text = article.text
        if len(text.strip()) < 20:
            raise HTTPException(status_code=400, detail="Could not extract text from URL.")
        result = detect_fake_news(text)
        result["article_title"] = article.title
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch article: {str(e)}")

