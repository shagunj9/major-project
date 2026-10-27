import pandas as pd
from config import OUTPUT_DIR

def create_evaluation_outputs(sentiment_rows, profiles):
    """
    Create descriptive diagnostics and a small annotation file.
    Automated accuracy cannot be claimed without human-annotated ground truth.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if sentiment_rows.empty:
        diagnostics = pd.DataFrame([{"Metric": "aspect_mentions", "Value": 0}])
    else:
        diagnostics = pd.DataFrame([
            {"Metric": "aspect_mentions", "Value": len(sentiment_rows)},
            {"Metric": "unique_products_with_aspects", "Value": sentiment_rows["ProductId"].nunique()},
            {"Metric": "unique_aspects", "Value": sentiment_rows["Aspect"].nunique()},
            {"Metric": "positive_fraction", "Value": (sentiment_rows["Sentiment"] == "positive").mean()},
            {"Metric": "negative_fraction", "Value": (sentiment_rows["Sentiment"] == "negative").mean()},
            {"Metric": "neutral_fraction", "Value": (sentiment_rows["Sentiment"] == "neutral").mean()},
            {"Metric": "mean_sentiment_model_confidence", "Value": sentiment_rows["SentimentConfidence"].mean()},
            {"Metric": "product_aspect_groups", "Value": len(profiles)},
        ])

    diagnostics.to_csv(OUTPUT_DIR / "evaluation_diagnostics.csv", index=False)

    # Annotate these rows manually to create a small ground-truth evaluation set.
    sample = sentiment_rows.sample(
        n=min(100, len(sentiment_rows)), random_state=42
    ).copy() if not sentiment_rows.empty else pd.DataFrame()

    if not sample.empty:
        sample = sample[["ProductId", "Sentence", "Aspect", "SubAspect", "Sentiment"]]
        sample = sample.rename(columns={"Sentiment": "PredictedSentiment"})
        sample["TrueAspect"] = ""
        sample["TrueSentiment"] = ""
        sample["IsAspectCorrect"] = ""
        sample.to_csv(OUTPUT_DIR / "manual_annotation_template.csv", index=False)

    print(f"Saved: {OUTPUT_DIR / 'evaluation_diagnostics.csv'}")
    if not sample.empty:
        print(f"Saved annotation template with {len(sample)} rows.")
        print("Fill TrueAspect, TrueSentiment and IsAspectCorrect before calculating accuracy.")
    return diagnostics
