from src.data_preprocessing import load_and_clean_reviews
from src.aspect_extraction import extract_aspect_mentions
from src.sentiment_analysis import analyse_aspect_sentiment
from src.evidence_analysis import build_evidence_profiles
from src.product_intelligence import build_product_intelligence
from src.summary_generation import generate_product_summaries
from src.evaluation import create_evaluation_outputs

def main():
    print("\nSTEP 1/7 — Load and clean reviews")
    reviews = load_and_clean_reviews()

    print("\nSTEP 2/7 — Extract aspects and sub-aspects")
    mentions = extract_aspect_mentions(reviews)
    if mentions.empty:
        print("No aspect mentions found. Inspect the taxonomy in src/aspect_extraction.py.")
        return

    print("\nSTEP 3/7 — Predict sentiment for aspect-containing sentences")
    sentiment_rows = analyse_aspect_sentiment(mentions)

    print("\nSTEP 4/7 — Build evidence profiles")
    profiles = build_evidence_profiles(sentiment_rows)

    print("\nSTEP 5/7 — Build product-level insights")
    filtered_profiles, product_scores = build_product_intelligence(profiles)

    print("\nSTEP 6/7 — Generate evidence-grounded summaries")
    summaries = generate_product_summaries(sentiment_rows, filtered_profiles)

    print("\nSTEP 7/7 — Create evaluation diagnostics")
    create_evaluation_outputs(sentiment_rows, profiles)

    print("\nPipeline complete.")
    print(f"Products scored: {len(product_scores):,}")
    print(f"Summaries generated: {len(summaries):,}")
    print("Check the outputs/ folder.")

if __name__ == "__main__":
    main()
