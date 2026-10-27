from pathlib import Path

# All paths are relative to this project's root folder.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# The Kaggle file is usually named Reviews.csv.
DATASET_PATH = DATA_DIR / "Reviews.csv"

# Start small so you can test the complete pipeline on your computer.
# Set to None to process every row (this may take a long time).
MAX_REVIEWS = 10000

# Hugging Face model for sentence-level positive/negative sentiment.
# This is a baseline model, not a food-specific aspect sentiment model.
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# The model is binary. Predictions below this confidence are marked neutral/uncertain.
SENTIMENT_CONFIDENCE_THRESHOLD = 0.60

# Minimum number of aspect mentions before a product-aspect profile is reported.
MIN_ASPECT_MENTIONS = 3

# Default shopper preference weights. These are examples and can be changed.
ASPECT_WEIGHTS = {
    "taste": 1.0,
    "texture": 1.0,
    "freshness": 1.0,
    "packaging": 1.0,
    "value": 1.0,
    "ingredients": 1.0,
    "other_food_quality": 1.0,
}
