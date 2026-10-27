import pandas as pd
from transformers import pipeline
from config import (
    OUTPUT_DIR, SENTIMENT_MODEL, SENTIMENT_CONFIDENCE_THRESHOLD
)

def _normalise_label(label, score):
    label = str(label).lower()
    if score < SENTIMENT_CONFIDENCE_THRESHOLD:
        return "neutral"
    if "positive" in label or label == "label_1":
        return "positive"
    if "negative" in label or label == "label_0":
        return "negative"
    return "neutral"

def analyse_aspect_sentiment(mentions, model_name=SENTIMENT_MODEL):
    """
    Baseline: predict sentiment for the sentence containing an aspect.
    Important limitation: the model is not explicitly conditioned on the aspect.
    """
    if mentions.empty:
        result = mentions.copy()
        result["Sentiment"] = pd.Series(dtype="object")
        result["SentimentConfidence"] = pd.Series(dtype="float")
        return result

    print(f"Loading sentiment model: {model_name}")
    classifier = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        truncation=True,
        device=-1,  # CPU; set to 0 only if a compatible CUDA GPU is configured.
    )

    texts = mentions["Sentence"].fillna("").astype(str).tolist()
    predictions = []
    batch_size = 16
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        predictions.extend(classifier(batch, batch_size=batch_size))

    result = mentions.copy()
    result["SentimentConfidence"] = [float(p["score"]) for p in predictions]
    result["Sentiment"] = [
        _normalise_label(p["label"], float(p["score"])) for p in predictions
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "aspect_sentiment.csv"
    result.to_csv(path, index=False)
    print(f"Saved: {path}")
    return result
