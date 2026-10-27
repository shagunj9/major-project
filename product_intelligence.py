import pandas as pd
from config import OUTPUT_DIR, ASPECT_WEIGHTS, MIN_ASPECT_MENTIONS

def build_product_intelligence(profiles):
    """Create product-level profiles and a transparent preference-weighted score."""
    if profiles.empty:
        return profiles.copy(), pd.DataFrame(columns=["ProductId", "ProductScore"])

    data = profiles.copy()
    data = data[data["MentionCount"] >= MIN_ASPECT_MENTIONS].copy()
    data["AspectWeight"] = data["Aspect"].map(ASPECT_WEIGHTS).fillna(1.0)

    # Confidence shrinks low-confidence sentiment toward zero; evidence count is
    # reported separately rather than treated as proof that sentiment is correct.
    data["WeightedEvidenceScore"] = (
        data["SentimentScore"] * data["MeanModelConfidence"] * data["AspectWeight"]
    )

    product_scores = data.groupby("ProductId").agg(
        ProductScore=("WeightedEvidenceScore", "mean"),
        SupportedAspectCount=("Aspect", "nunique"),
        TotalAspectMentions=("MentionCount", "sum"),
    ).reset_index().sort_values("ProductScore", ascending=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_DIR / "product_aspect_profiles.csv", index=False)
    product_scores.to_csv(OUTPUT_DIR / "product_scores.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'product_aspect_profiles.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'product_scores.csv'}")
    return data, product_scores
