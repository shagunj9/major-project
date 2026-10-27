import re
import pandas as pd
from config import OUTPUT_DIR

# Initial interpretable taxonomy. Extend it after inspecting real review examples.
ASPECT_TERMS = {
    "taste": {
        "flavour": ["taste", "flavor", "flavour", "tasty", "delicious", "bland",
                    "bitter", "sweet", "salty", "spicy", "sour", "aftertaste"],
        "sweetness": ["sweet", "sugary", "too sweet", "sugar"],
    },
    "texture": {
        "crunchiness": ["crunchy", "crunch", "crispy", "crisp"],
        "consistency": ["texture", "smooth", "grainy", "lumpy", "watery", "thick"],
        "softness": ["soft", "hard", "chewy", "tough"],
    },
    "freshness": {
        "freshness": ["fresh", "stale", "spoiled", "expired", "expiry", "rancid"],
    },
    "packaging": {
        "container_and_seal": ["package", "packaging", "container", "seal", "sealed",
                               "lid", "box", "bag", "bottle"],
        "shipping_damage": ["broken", "leaking", "leaked", "damaged", "spill", "spilled"],
    },
    "value": {
        "price": ["price", "expensive", "cheap", "cost", "overpriced", "affordable"],
        "quantity_for_price": ["value", "worth", "quantity", "size", "small amount"],
    },
    "ingredients": {
        "ingredient_quality": ["ingredient", "ingredients", "organic", "artificial",
                               "preservative", "additive", "gluten", "sodium"],
    },
    "other_food_quality": {
        "preparation_and_use": ["easy to make", "easy to prepare", "mix", "dissolve",
                                "brew", "cook", "prepare"],
        "aroma": ["smell", "aroma", "fragrant", "odor", "odour"],
    },
}

def split_sentences(text):
    """Simple sentence splitter; can later be replaced with spaCy or NLTK."""
    text = str(text).replace("\n", " ")
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def extract_aspect_mentions(reviews):
    """
    Extract aspect mentions using a transparent lexicon baseline.
    Output has one row per review sentence and matched sub-aspect.
    """
    rows = []
    for row in reviews.itertuples(index=False):
        text = getattr(row, "Text", "")
        for sentence in split_sentences(text):
            lower = sentence.lower()
            for aspect, subaspects in ASPECT_TERMS.items():
                for subaspect, terms in subaspects.items():
                    matched = [term for term in terms if term in lower]
                    if matched:
                        rows.append({
                            "ProductId": row.ProductId,
                            "Score": row.Score,
                            "ReviewDate": getattr(row, "ReviewDate", pd.NaT),
                            "ReviewYear": getattr(row, "ReviewYear", None),
                            "HelpfulnessNumerator": row.HelpfulnessNumerator,
                            "HelpfulnessDenominator": row.HelpfulnessDenominator,
                            "Summary": getattr(row, "Summary", ""),
                            "Sentence": sentence,
                            "Aspect": aspect,
                            "SubAspect": subaspect,
                            "MatchedTerms": "|".join(sorted(set(matched))),
                        })

    columns = [
        "ProductId", "Score", "ReviewDate", "ReviewYear",
        "HelpfulnessNumerator", "HelpfulnessDenominator", "Summary",
        "Sentence", "Aspect", "SubAspect", "MatchedTerms"
    ]
    mentions = pd.DataFrame(rows, columns=columns)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "aspect_mentions.csv"
    mentions.to_csv(path, index=False)
    print(f"Aspect mentions extracted: {len(mentions):,}")
    print(f"Saved: {path}")
    return mentions
