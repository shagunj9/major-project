import numpy as np
import pandas as pd
from config import OUTPUT_DIR, MIN_ASPECT_MENTIONS

def build_evidence_profiles(sentiment_rows):
    """Aggregate aspect evidence while keeping volume and confidence separate."""
    columns = [
        "ProductId", "Aspect", "SubAspect", "MentionCount", "PositiveCount",
        "NegativeCount", "NeutralCount", "MeanModelConfidence",
        "MeanHelpfulnessRatio", "SentimentScore", "EvidenceLevel"
    ]
    if sentiment_rows.empty:
        return pd.DataFrame(columns=columns)

    df = sentiment_rows.copy()
    df["HelpfulnessRatio"] = np.where(
        df["HelpfulnessDenominator"] > 0,
        df["HelpfulnessNumerator"] / df["HelpfulnessDenominator"],
        np.nan
    )
    df["SentimentValue"] = df["Sentiment"].map({
        "positive": 1.0, "neutral": 0.0, "negative": -1.0
    }).fillna(0.0)

    group_cols = ["ProductId", "Aspect", "SubAspect"]
    profiles = df.groupby(group_cols, dropna=False).agg(
        MentionCount=("Sentence", "size"),
        PositiveCount=("Sentiment", lambda x: int((x == "positive").sum())),
        NegativeCount=("Sentiment", lambda x: int((x == "negative").sum())),
        NeutralCount=("Sentiment", lambda x: int((x == "neutral").sum())),
        MeanModelConfidence=("SentimentConfidence", "mean"),
        MeanHelpfulnessRatio=("HelpfulnessRatio", "mean"),
        SentimentScore=("SentimentValue", "mean"),
    ).reset_index()

    profiles["EvidenceLevel"] = profiles["MentionCount"].apply(
        lambda n: "higher" if n >= 20 else ("moderate" if n >= MIN_ASPECT_MENTIONS else "low")
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "product_aspect_evidence.csv"
    profiles.to_csv(path, index=False)
    print(f"Saved: {path}")
    return profiles
