import re
import pandas as pd
from config import DATASET_PATH, OUTPUT_DIR, MAX_REVIEWS

REQUIRED_COLUMNS = ["ProductId", "Score", "Time", "HelpfulnessNumerator",
                    "HelpfulnessDenominator", "Summary", "Text"]

def load_and_clean_reviews(dataset_path=DATASET_PATH, max_reviews=MAX_REVIEWS):
    """Load the Kaggle CSV, validate columns, and perform conservative cleaning."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {dataset_path}\n"
            "Download Reviews.csv from the Amazon Fine Food Reviews Kaggle dataset "
            "and place it inside the project's data/ folder."
        )

    df = pd.read_csv(dataset_path, low_memory=False)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}. "
            f"Columns found: {list(df.columns)}"
        )

    df = df.copy()
    for col in ["Text", "Summary", "ProductId"]:
        df[col] = df[col].fillna("").astype(str)

    df["Text"] = df["Text"].map(lambda x: re.sub(r"\s+", " ", x).strip())
    df["Summary"] = df["Summary"].map(lambda x: re.sub(r"\s+", " ", x).strip())
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["HelpfulnessNumerator"] = pd.to_numeric(
        df["HelpfulnessNumerator"], errors="coerce"
    ).fillna(0)
    df["HelpfulnessDenominator"] = pd.to_numeric(
        df["HelpfulnessDenominator"], errors="coerce"
    ).fillna(0)

    df = df[(df["Text"].str.len() > 0) & (df["ProductId"].str.len() > 0)]
    df = df[df["Score"].between(1, 5)]
    df = df.drop_duplicates(subset=["ProductId", "Text"]).reset_index(drop=True)

    # Dataset Time is normally a Unix timestamp in seconds.
    df["ReviewDate"] = pd.to_datetime(df["Time"], unit="s", errors="coerce")
    df["ReviewYear"] = df["ReviewDate"].dt.year

    if max_reviews is not None and len(df) > max_reviews:
        # Fixed random_state makes the development sample reproducible.
        df = df.sample(n=max_reviews, random_state=42).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "cleaned_reviews.csv"
    df.to_csv(output_path, index=False)

    print(f"Cleaned reviews: {len(df):,}")
    print(f"Unique products: {df['ProductId'].nunique():,}")
    print(f"Saved: {output_path}")
    return df
