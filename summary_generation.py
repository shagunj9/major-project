import pandas as pd
from config import OUTPUT_DIR

def generate_product_summaries(sentiment_rows, profiles, max_products=100):
    """
    Produce evidence-grounded template summaries.
    This first version does not use an LLM, which keeps claims traceable to data.
    """
    if profiles.empty or sentiment_rows.empty:
        return pd.DataFrame(columns=["ProductId", "Summary"])

    chosen_products = profiles.groupby("ProductId")["MentionCount"].sum()
    chosen_products = chosen_products.sort_values(ascending=False).head(max_products).index

    summaries = []
    for product_id in chosen_products:
        p = profiles[profiles["ProductId"] == product_id].copy()
        r = sentiment_rows[sentiment_rows["ProductId"] == product_id].copy()
        p = p.sort_values(["MentionCount", "SentimentScore"], ascending=[False, False])

        positive_aspects = p[p["SentimentScore"] > 0.15].head(3)
        negative_aspects = p[p["SentimentScore"] < -0.15].sort_values(
            "SentimentScore"
        ).head(3)

        positive_text = ", ".join(
            f"{row.Aspect} ({int(row.MentionCount)} mentions)"
            for row in positive_aspects.itertuples()
        ) or "No clearly positive aspect signal in the current sample"

        negative_text = ", ".join(
            f"{row.Aspect} ({int(row.MentionCount)} mentions)"
            for row in negative_aspects.itertuples()
        ) or "No clearly negative aspect signal in the current sample"

        evidence_sentences = []
        for aspect in list(positive_aspects["Aspect"]) + list(negative_aspects["Aspect"]):
            matching = r[r["Aspect"] == aspect]
            if not matching.empty:
                # Use the most confident sentence as an example, not as proof of consensus.
                example = matching.sort_values("SentimentConfidence", ascending=False).iloc[0]
                evidence_sentences.append(f"{aspect}: “{example['Sentence']}”")
            if len(evidence_sentences) >= 3:
                break

        summaries.append({
            "ProductId": product_id,
            "PositiveSignals": positive_text,
            "Concerns": negative_text,
            "EvidenceExamples": " | ".join(evidence_sentences),
            "TotalAspectMentions": int(p["MentionCount"].sum()),
            "Caveat": "Signals reflect the processed sample and automatic predictions; verify source reviews.",
        })

    result = pd.DataFrame(summaries)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_DIR / "product_summaries.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'product_summaries.csv'}")
    return result
