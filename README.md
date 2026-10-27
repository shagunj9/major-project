# Aspect-Based Review Analysis and Intelligent Summary Generation

This project analyses the Amazon Fine Food Reviews dataset. It extracts food-related aspects, predicts sentence-level sentiment, aggregates product evidence, creates product profiles, and produces template-based summaries.

## Project structure

- `main.py`: runs the full pipeline.
- `config.py`: paths, sample size, model name, thresholds and aspect weights.
- `src/data_preprocessing.py`: loads and cleans the Kaggle CSV.
- `src/aspect_extraction.py`: rule/lexicon-based aspect and sub-aspect extraction.
- `src/sentiment_analysis.py`: Hugging Face sentence-level sentiment baseline.
- `src/evidence_analysis.py`: aggregates evidence counts, confidence and helpfulness.
- `src/product_intelligence.py`: product profiles and weighted scores.
- `src/summary_generation.py`: traceable template summaries.
- `src/evaluation.py`: diagnostics and a manual annotation template.

## Setup in VS Code

1. Download `Reviews.csv` from https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
2. Put the file in this project's `data/` folder.
3. Open the project folder in VS Code.
4. Create a virtual environment in the VS Code terminal:

   Windows:
   ```powershell
   py -m venv .venv
   .venv\Scripts\activate
   ```

   If PowerShell blocks activation, use Command Prompt or run `.venv\Scripts\python.exe` directly.

5. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
6. Run:
   ```powershell
   python main.py
   ```

The first sentiment-analysis run downloads the pretrained Hugging Face model and may take time, especially on CPU.

## Important limitations

- The initial aspect taxonomy is lexicon/rule-based. It will miss aspects not listed in `src/aspect_extraction.py` and can match words in irrelevant contexts.
- The pretrained sentiment model predicts sentiment for the whole sentence containing an aspect; it is **not explicitly conditioned on each aspect**. A sentence with both praise and criticism can therefore be misclassified for one of its aspects.
- The sentiment model is a general-domain baseline, not a food-review-specific validated ABSA model.
- The dataset has no gold aspect labels. The diagnostics are not accuracy measurements. Manually annotate `outputs/manual_annotation_template.csv` and use a held-out labelled set to report precision, recall and F1.
- Helpfulness votes are not proof of truth. Model confidence is not the same as correctness.
- The default `MAX_REVIEWS = 10000` is for initial testing. Set it to `None` in `config.py` to process the entire dataset after verifying runtime and memory.
- Product scores are experimental ranking signals, not objective product quality ratings.
- Template summaries are deliberately used first so that claims remain traceable to source sentences. LLM-based generation can be added and evaluated later.

## Generated outputs

The `outputs/` folder is populated when the pipeline runs:
- `cleaned_reviews.csv`
- `aspect_mentions.csv`
- `aspect_sentiment.csv`
- `product_aspect_evidence.csv`
- `product_aspect_profiles.csv`
- `product_scores.csv`
- `product_summaries.csv`
- `evaluation_diagnostics.csv`
- `manual_annotation_template.csv`

Do not commit the full dataset or generated files to a public Git repository without checking file size and dataset terms. For a Kaggle notebook, attach the dataset through Kaggle's dataset interface instead of uploading a duplicate.
