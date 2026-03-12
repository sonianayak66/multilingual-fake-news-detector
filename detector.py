import os, json, requests, torch
import numpy as np
from langdetect import detect
from groq import Groq
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

load_dotenv()
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
groq_client    = Groq(api_key=GROQ_API_KEY)

# ── LAYER 1: Load XLM-RoBERTa ─────────────────────────
print("Loading XLM-RoBERTa model from HuggingFace...")
MODEL_NAME = "Sonia66/multilingual-fake-news-detector"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
nlp_model  = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
nlp_model.eval()
print("✅ XLM-RoBERTa loaded!")

# ── LIME Explainer ─────────────────────────────────────
explainer = LimeTextExplainer(class_names=["REAL", "FAKE"])
print("✅ Fake News Detector ready — 3 layers + LIME active!")

# ── LAYER 1: XLM-RoBERTa inference ────────────────────
def xlm_roberta_analyze(text):
    try:
        inputs = tokenizer(
            text[:512],
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        )
        with torch.no_grad():
            outputs   = nlp_model(**inputs)
            probs     = torch.softmax(outputs.logits, dim=1)
            fake_prob = probs[0][1].item()
            real_prob = probs[0][0].item()

        verdict    = "FAKE" if fake_prob > 0.5 else "REAL"
        confidence = fake_prob if verdict == "FAKE" else real_prob
        return {
            "verdict":    verdict,
            "confidence": round(confidence * 100, 2),
            "fake_prob":  round(fake_prob * 100, 2),
            "real_prob":  round(real_prob * 100, 2)
        }
    except Exception as e:
        print(f"XLM-RoBERTa error: {e}")
        return {"verdict": "UNKNOWN", "confidence": 50.0, "fake_prob": 50.0, "real_prob": 50.0}

# ── LIME Explainability ────────────────────────────────
def get_lime_explanation(text):
    try:
        def predict_proba(texts):
            results = []
            for t in texts:
                inputs = tokenizer(
                    t[:512],
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                    padding=True
                )
                with torch.no_grad():
                    outputs = nlp_model(**inputs)
                    probs   = torch.softmax(outputs.logits, dim=1)
                results.append(probs[0].numpy())
            return np.array(results)

        exp = explainer.explain_instance(
            text[:512],
            predict_proba,
            num_features=8,
            num_samples=100
        )
        lime_words = exp.as_list(label=1)
        explanation = [
            {"word": word, "contribution": round(float(score), 4)}
            for word, score in lime_words
        ]
        explanation.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return explanation
    except Exception as e:
        print(f"LIME error: {e}")
        return []

# ── LAYER 2: Groq Llama-70B ────────────────────────────
def groq_analyze(text, lang):
    try:
        prompt = f"""You are an expert fake news detector with 10 years experience.
Analyze this article and determine if it is FAKE or REAL news.

FAKE news indicators:
- Sensational or clickbait headlines
- No credible sources cited
- Emotional manipulation language
- Conspiracy theories
- Unverified extraordinary claims
- Poor grammar or spelling
- Anonymous or unknown authors

REAL news indicators:
- Credible sources cited (Reuters, AP, BBC etc.)
- Balanced reporting
- Verifiable facts and dates
- Professional journalistic language
- Named journalists and publications

Language of article: {lang}

Return ONLY valid JSON in this exact format:
{{
    "verdict": "FAKE" or "REAL",
    "confidence": 0.95,
    "red_flags": ["flag1", "flag2"],
    "reasoning": "One sentence explanation"
}}

Article: {text[:2000]}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw   = response.choices[0].message.content.strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        data  = json.loads(raw[start:end])
        return {
            "verdict":    data.get("verdict", "UNKNOWN"),
            "confidence": round(float(data.get("confidence", 0.5)) * 100, 2),
            "red_flags":  data.get("red_flags", []),
            "reasoning":  data.get("reasoning", "")
        }
    except Exception as e:
        print(f"Groq error: {e}")
        return {"verdict": "UNKNOWN", "confidence": 50.0, "red_flags": [], "reasoning": "Analysis failed"}

# ── LAYER 3: Google Fact Check ─────────────────────────
def fact_check(text):
    try:
        url    = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {"query": text[:200], "key": GOOGLE_API_KEY}
        r      = requests.get(url, params=params, timeout=5)
        data   = r.json()
        claims = data.get("claims", [])
        if not claims:
            return {"found": False, "score": 50.0, "sources": []}
        sources = []
        fake_count, real_count = 0, 0
        for claim in claims[:3]:
            for review in claim.get("claimReview", []):
                rating = review.get("textualRating", "").upper()
                source = review.get("publisher", {}).get("name", "Unknown")
                sources.append(f"{source}: {review.get('textualRating', '')}")
                if any(w in rating for w in ["FALSE", "FAKE", "MISLEADING", "WRONG"]):
                    fake_count += 1
                elif any(w in rating for w in ["TRUE", "CORRECT", "ACCURATE"]):
                    real_count += 1
        if fake_count > real_count:
            score = 85.0
        elif real_count > fake_count:
            score = 15.0
        else:
            score = 50.0
        return {"found": True, "score": score, "sources": sources[:3]}
    except Exception as e:
        print(f"Fact check error: {e}")
        return {"found": False, "score": 50.0, "sources": []}

# ── ENSEMBLE: Combine all 3 layers ────────────────────
def detect_fake_news(text):
    lang = "unknown"
    try:
        lang = detect(text)
    except:
        pass

    # Run all 3 layers
    layer1 = xlm_roberta_analyze(text)
    layer2 = groq_analyze(text, lang)
    layer3 = fact_check(text)

    # Convert to fake probability scores
    l1_fake_score = layer1["fake_prob"]
    l2_fake_score = layer2["confidence"] if layer2["verdict"] == "FAKE" else (100 - layer2["confidence"])

    # Weighted ensemble
    if layer3["found"]:
        l3_fake_score = layer3["score"]
        final_score   = (l1_fake_score * 0.35) + (l2_fake_score * 0.50) + (l3_fake_score * 0.15)
    else:
        final_score = (l1_fake_score * 0.35) + (l2_fake_score * 0.65)

    verdict    = "FAKE" if final_score > 50 else "REAL"
    confidence = round(final_score if verdict == "FAKE" else (100 - final_score), 2)

    # LIME explanation
    lime_explanation = get_lime_explanation(text)

    return {
        "verdict":            verdict,
        "confidence":         confidence,
        "language":           lang,
        "red_flags":          layer2["red_flags"],
        "explanation":        layer2["reasoning"],
        "fact_check_sources": layer3["sources"],
        "lime_explanation":   lime_explanation,
        "layer_scores": {
            "xlm_roberta":    round(l1_fake_score, 2),
            "groq_llama_70b": round(l2_fake_score, 2),
            "fact_check":     round(layer3["score"], 2)
        }
    }