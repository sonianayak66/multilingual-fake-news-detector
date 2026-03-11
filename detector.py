import os
import json
import requests
from langdetect import detect
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

print("✅ Fake News Detector ready!")


def groq_analyze(text: str, lang: str) -> dict:
    try:
        prompt = f"""You are an expert fake news detector with 10 years experience fact-checking for major news organizations.

Your job: Analyze the article below and determine if it is FAKE or REAL news.

FAKE news indicators:
- Sensational or exaggerated claims ("cures all", "overnight", "they don't want you to know")
- Conspiracy language ("hiding", "secret", "cover-up", "before they delete")
- Missing sources or unverifiable claims
- Emotional manipulation language
- Clickbait headlines
- Anonymous or unknown sources

REAL news indicators:
- Specific verifiable facts (dates, names, locations)
- Quotes from named sources
- Measured language without exaggeration
- References to official organizations
- Consistent with known facts

Language code of article: {lang}

Be very precise. If the article has verifiable facts and named sources, confidence for REAL should be 0.90+.
If the article has conspiracy language and unverifiable claims, confidence for FAKE should be 0.90+.

Return ONLY this exact JSON, no extra text:
{{
  "verdict": "FAKE" or "REAL" or "UNCERTAIN",
  "confidence": a number between 0.0 and 1.0,
  "red_flags": ["list each suspicious phrase found, empty list if REAL"],
  "reasoning": "one clear sentence explaining your verdict"
}}

Article to analyze:
{text[:2000]}
"""
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw.strip())

    except Exception as e:
        import traceback
        print(f"Groq layer error: {e}")
        traceback.print_exc()
        return {
            "verdict": "UNCERTAIN",
            "confidence": 0.5,
            "red_flags": [],
            "reasoning": "Analysis unavailable"
        }


def fact_check(text: str) -> dict:
    try:
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            "query": text[:200],
            "key": GOOGLE_API_KEY
        }
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        claims = data.get("claims", [])
        if claims:
            return {
                "found": True,
                "score": 0.9,
                "sources": [
                    c.get("claimReview", [{}])[0].get("publisher", {}).get("name", "Unknown")
                    for c in claims[:3]
                ]
            }
        return {"found": False, "score": 0.5, "sources": []}
    except Exception as e:
        print(f"Fact check layer error: {e}")
        return {"found": False, "score": 0.5, "sources": []}


def detect_fake_news(text: str) -> dict:
    print(f"\n🔍 Analyzing article ({len(text)} characters)...")

    try:
        lang = detect(text)
    except:
        lang = "unknown"
    print(f"🌐 Detected language: {lang}")

    print("🧠 Running Groq/Llama-70B reasoning...")
    groq_result = groq_analyze(text, lang)

    print("📰 Running fact check...")
    fact_result = fact_check(text)

    groq_confidence = groq_result["confidence"]
    fact_score = fact_result["score"]
    groq_verdict = groq_result["verdict"]

    # If fact check found a match, use it to boost or penalize
    if fact_result["found"]:
        # Fact check found → blend 80% Groq + 20% fact check
        final_confidence = (groq_confidence * 0.80) + (fact_score * 0.20)
    else:
        # No fact check found → trust Groq 100%
        final_confidence = groq_confidence

    # Verdict comes directly from Groq
    verdict = groq_verdict if groq_verdict != "UNCERTAIN" else (
        "FAKE" if final_confidence > 0.5 else "REAL"
    )

    return {
        "verdict": verdict,
        "confidence": round(final_confidence * 100, 1),
        "language": lang,
        "red_flags": groq_result.get("red_flags", []),
        "explanation": groq_result.get("reasoning", ""),
        "fact_check_sources": fact_result.get("sources", []),
        "layer_scores": {
            "groq_llama_70b": round(groq_confidence * 100, 1),
            "fact_check":     round(fact_score * 100, 1)
        }
    }