# SentimentIQ

**A leakage-safe, three-class sentiment analysis platform that benchmarks six models on English product reviews, with cross-domain evaluation and a seven-page Streamlit app.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange) ![PyTorch](https://img.shields.io/badge/PyTorch-optional-red) ![Streamlit](https://img.shields.io/badge/Streamlit-app-ff4b4b) ![Tests](https://img.shields.io/badge/tests-35%20passing-brightgreen)

SentimentIQ classifies a review as **Positive**, **Neutral** or **Negative** from its text alone. It compares classical machine learning (Multinomial Naive Bayes, Logistic Regression, Linear SVM) with PyTorch networks (MLP, TextCNN, BiLSTM) on two English review datasets, and measures how well each model transfers from one domain to the other. Every number in this README is written from the saved files in `reports/` by `scripts/update_readme_results.py`; none is typed by hand.

## Table of contents
1. [Highlights](#highlights)
2. [Datasets and labels](#datasets-and-labels)
3. [Project architecture](#project-architecture)
4. [Methodology](#methodology)
5. [Models](#models)
6. [Results](#results)
7. [Cross-domain evaluation](#cross-domain-evaluation)
8. [Robustness challenge](#robustness-challenge)
9. [Streamlit application](#streamlit-application)
10. [Installation and usage](#installation-and-usage)
11. [Testing](#testing)
12. [Troubleshooting PyTorch on Windows](#troubleshooting-pytorch-on-windows)
13. [Limitations](#limitations)
14. [Future work](#future-work)
15. [Project structure](#project-structure)

## Highlights
- **Six models, one interface.** Naive Bayes, Logistic Regression, Linear SVM, MLP, TextCNN and BiLSTM are trained, evaluated and served through the same code path.
- **Leakage prevention by design.** Only `review_text` is a model input; exact and near-duplicates are removed before and after splitting; vectorisers and vocabularies are fitted on training data only; the test split is scored once.
- **Honest evaluation.** Macro-F1 is the headline metric on an imbalanced task; each transfer result is shown next to an in-domain baseline; limitations are stated, not hidden.
- **Cross-domain tests.** Amazon to Clothing, Clothing to Amazon, and train-on-both evaluated on held-out reviews.
- **Product-ready app.** Live prediction with explanations, bulk CSV analysis with trend and per-location insights, model evaluation, error analysis and cross-domain pages. The app never trains.
- **Resilient.** A missing dataset file or a broken PyTorch install does not crash the app.

## Datasets and labels
| Dataset | Source | Language |
|---|---|---|
| Amazon Fine Food | local CSV read directly from disk by path (never uploaded) | English |
| Women's Clothing | `data/processed/womens_clothing_reviews_cleaned.csv` | English |

Class distribution, duplicate counts and the leakage audit (generated from `reports/three_class/dataset_audit.json`):

<!-- RESULTS:AUDIT:START -->
| Dataset | Rows loaded | Exact duplicates removed | Rows after de-dup | Negative | Neutral | Positive |
|---|---|---|---|---|---|---|
| product_food | 29,847 | 0 | 29,847 | 4,309 | 2,262 | 23,276 |
| product_clothing | 22,634 | 1 | 22,633 | 2,369 | 2,823 | 17,441 |

Splits: train 36,735 / validation 7,869 / test 7,872. Identical texts shared by the two datasets: 0. Near-duplicates between the datasets removed: 0. Near-duplicates of training reviews removed from validation/test: {'product_food': {'validation': 2, 'test': 0}, 'product_clothing': {'validation': 1, 'test': 1}}. Rule: TF-IDF cosine >= 0.9 on reviews with >= 6 tokens; shorter reviews are compared by exact match only.
<!-- RESULTS:AUDIT:END -->

**Weak labels.** Labels are derived from star ratings: **1-2 stars Negative, 3 stars Neutral, 4-5 stars Positive.** They are *weak* labels, not hand-verified sentiment: a 3-star review can be glowing and a 5-star review can contain complaints. Scores therefore measure how well text predicts a rating bucket. Neutral is the rarest and noisiest class. Nothing is relabelled or translated.

**Amazon sample.** The reported results use a 50,000-row sample of the Amazon data, capped at about 30,000 rows for training, not the full ~197 MB file.

### Local Amazon dataset configuration
The Amazon file is read with Pandas straight from your disk. The path is resolved in this order:
1. environment variable `SENTIMENTIQ_AMAZON_PATH`
2. `local_paths.json` in the project root (git-ignored; written by the "Save path" button on the Home page)
3. default `C:\Users\HP\Downloads\amazon_fine_food_reviews_cleaned.csv`

The loader checks that the file exists, shows a message instead of crashing if it does not (the Clothing data keeps working), caches the result, and opens the file read-only. Files over 20 MB are read in 100,000-row chunks with a two-pass **stratified** sample (pass 1 counts classes; pass 2 keeps each row with probability `target / count`, seeded, so the sample is reproducible). The default cap is 30,000 rows; change it with `SENTIMENTIQ_SAMPLE_PRODUCT_FOOD`, the Home page, or `--sample product_food=N`. The 50 MB Streamlit upload limit applies only to the Bulk page.

## Project architecture
```
CSV files ──► datasets.py ──► three_class.py ──► training.py ──► saved models
 (2 domains)   load, label,    near-duplicates,    scikit-learn      models/three_class/
               de-duplicate,   stratified           and PyTorch
               stratified      70/15/15 split       models
               sample          + leakage audit
                                                         │
 reports/three_class/ ◄── evaluate.py, cross_domain.py ◄─┘
        │                 robustness_test.py
        ▼
 Streamlit app (reads saved models and reports; never trains)
```

## Methodology
**Preprocessing.** Text is lower-cased with whitespace, URLs, curly quotes and stretched characters normalised, and apostrophe-less contractions repaired (`wasnt` becomes `wasn't`). Stop words are **kept**, because words such as "not" and "but" carry most negative signal.

**Features.**
- Naive Bayes, Logistic Regression, Linear SVM: TF-IDF (sublinear tf, word 1-2 grams, apostrophes preserved), fitted on the training split only.
- MLP: TF-IDF with up to 15,000 features, fitted on the training split only.
- TextCNN and BiLSTM: a vocabulary built from training text only (minimum frequency 2, at most 30,000 words, `<unk>` for unseen words), sequences truncated to 160 tokens.

**Splitting.** Exact duplicates are removed first; each dataset is split with a stratified 70/15/15 train/validation/test split and the splits are pooled. Near-duplicates (TF-IDF cosine similarity of 0.90 or more on reviews of at least 6 tokens) are removed between the two datasets, and near-duplicates of a training review are removed from validation and test.

**Training and model selection.** The three classical models are grid-searched on validation macro-F1 using a `PredefinedSplit`, then fitted on the training split only. The networks use Adam, inverse-frequency class weights, gradient clipping and early stopping on validation loss (patience 3) with best-checkpoint restore. The best model is chosen on **validation** macro-F1; the test split is evaluated once.

**Metrics.** Accuracy, macro and weighted precision, recall and F1, per-class scores, a 3x3 confusion matrix, train/validation/test gaps, training time and inference time.

**Leakage safeguards.**
- `review_text` is the only input; `rating`, `review_id`, `source` and `domain` are listed in `FORBIDDEN_FEATURES` and tests check this.
- `assert_disjoint_splits` raises if any normalised text appears in two splits.
- Held-out-domain experiments assert the test domain never appears in training or validation data.

## Models
| Model | Input | Notes |
|---|---|---|
| Multinomial Naive Bayes | TF-IDF | smoothing and class prior tuned on validation macro-F1 |
| Logistic Regression | TF-IDF | grid-searched; balanced class weights |
| Linear SVM | TF-IDF | grid-searched; balanced class weights; outputs decision scores, not probabilities |
| MLP | TF-IDF, then 256 and 128 units, dropout 0.3 and 0.2 | PyTorch |
| TextCNN | learned embeddings, kernels 3/4/5 with 100 filters each, dropout 0.5 | PyTorch |
| BiLSTM | learned embeddings, bidirectional LSTM (hidden 96), dropout 0.4 | PyTorch |

PyTorch is **optional at run time**: if it cannot be imported, the app still opens and the classical models work.

## Results
<!-- RESULTS:THREE:START -->
| Model | Family | Train acc | Val macro-F1 | Test acc | Test macro-F1 | Train time (s) | Inference (ms/review) | Params |
|---|---|---|---|---|---|---|---|---|
| Multinomial Naive Bayes | Traditional ML | 0.9495 | 0.6498 | 0.8318 | 0.6445 | 75.8 | 0.255 | - |
| Logistic Regression ★ | Traditional ML | 0.9556 | 0.6636 | 0.8206 | 0.6519 | 78.7 | 0.240 | - |
| Linear SVM | Traditional ML | 0.9924 | 0.6518 | 0.8454 | 0.6475 | 43.6 | 0.246 | - |
| MLP (TF-IDF) | Neural network | 0.8483 | 0.6300 | 0.7730 | 0.6354 | 31.9 | 0.132 | 3873539 |
| TextCNN | Neural network | 0.8316 | 0.6000 | 0.7528 | 0.6023 | 409.7 | 0.380 | 2715443 |
| BiLSTM | Neural network | 0.8263 | 0.5899 | 0.7528 | 0.5993 | 931.4 | 0.311 | 2734787 |

★ = best by **validation** macro-F1. Split sizes: train 36,735, validation 7,869, test 7,872. Fit mode: `train_only`. Single split, single seed.

Per-domain test performance of the best model (Logistic Regression):

| Domain | Reviews | Accuracy | Macro-F1 |
|---|---|---|---|
| product_clothing | 3,394 | 0.7994 | 0.6305 |
| product_food | 4,478 | 0.8368 | 0.6502 |
<!-- RESULTS:THREE:END -->

Macro-F1 is the metric to read: the networks use class weights, which trades accuracy for minority-class recall. This is a single split with a single seed, so small gaps between models are within noise.

## Cross-domain evaluation
Three experiments are run with hyper-parameters fixed in advance (nothing is tuned on the held-out data): train on Amazon and test on Clothing, train on Clothing and test on Amazon, and train on both and score the held-out test reviews of both. Each transfer result is shown beside an in-domain baseline. The saved runs used at most 6,000 training and 3,000 test rows and 5 epochs for the networks, so they are smaller than the main results. They describe food-versus-clothing transfer only.

<!-- RESULTS:XD:START -->
**Transfer between datasets** (trained on one, tested on the other):

| Train -> test | Model | Accuracy | Macro-F1 | In-domain baseline macro-F1 | Test OOV rate |
|---|---|---|---|---|---|
| Amazon -> Clothing | naive_bayes | 0.7057 | 0.4934 | 0.6149 | 7.9% |
| Amazon -> Clothing | logistic_regression | 0.7043 | 0.5269 | 0.6097 | 7.9% |
| Amazon -> Clothing | linear_svm | 0.7213 | 0.4941 | 0.5902 | 7.9% |
| Amazon -> Clothing | mlp | 0.6063 | 0.5299 | 0.6251 | 7.9% |
| Amazon -> Clothing | textcnn | 0.5630 | 0.4651 | 0.5360 | 7.9% |
| Amazon -> Clothing | bilstm | 0.5267 | 0.4426 | 0.5296 | 7.9% |
| Clothing -> Amazon | naive_bayes | 0.6587 | 0.5064 | 0.5568 | 17.4% |
| Clothing -> Amazon | logistic_regression | 0.6683 | 0.5282 | 0.6213 | 17.4% |
| Clothing -> Amazon | linear_svm | 0.6973 | 0.4938 | 0.5980 | 17.4% |
| Clothing -> Amazon | mlp | 0.4753 | 0.4480 | 0.6162 | 17.4% |
| Clothing -> Amazon | textcnn | 0.3983 | 0.3657 | 0.4675 | 17.4% |
| Clothing -> Amazon | bilstm | 0.4883 | 0.4005 | 0.4775 | 17.4% |

**Both datasets pooled**, scored on the held-out test splits of both (not a transfer test):

| Model | Accuracy | Macro-F1 | Amazon macro-F1 | Clothing macro-F1 |
|---|---|---|---|---|
| naive_bayes | 0.8330 | 0.5863 | 0.5639 | 0.5633 |
| logistic_regression | 0.8072 | 0.6046 | 0.6014 | 0.5922 |
| linear_svm | 0.8276 | 0.5835 | 0.5725 | 0.5808 |
| mlp | 0.7829 | 0.6070 | 0.5919 | 0.6010 |
| textcnn | 0.6949 | 0.5267 | 0.5155 | 0.5164 |
| bilstm | 0.6007 | 0.4564 | 0.4420 | 0.4625 |
<!-- RESULTS:XD:END -->

## Robustness challenge
<!-- RESULTS:ROB:START -->
39 hand-written examples (diagnostic only):

| Model | Accuracy | Macro-F1 | Accuracy (clear labels only) |
|---|---|---|---|
| naive_bayes | 0.667 | 0.636 | 0.733 |
| logistic_regression | 0.564 | 0.557 | 0.567 |
| linear_svm | 0.590 | 0.579 | 0.600 |
| mlp | 0.538 | 0.533 | 0.533 |
| textcnn | 0.564 | 0.560 | 0.567 |
| bilstm | 0.487 | 0.467 | 0.567 |
<!-- RESULTS:ROB:END -->

These are hand-written reviews (sarcasm, negation, typos, ambiguity) outside every split. With so few examples this is a diagnostic probe and does **not** prove real-world generalisation.

## Streamlit application
| Page | What it does |
|---|---|
| Home | Dataset statistics for both domains, model status, local Amazon path settings |
| Live Sentiment Analysis | One review, any model, class probabilities (decision scores for Linear SVM) and the words that influenced the prediction |
| Bulk Sentiment Analysis | Upload a CSV with a review-text column, download predictions; optional date and branch/store/location columns unlock a trend chart, a per-location table and plain-language insights |
| Model Evaluation | Comparison table, confusion matrices, per-class scores, learning curves |
| Cross-Domain Evaluation | Saved transfer and pooled results; run a new experiment on demand |
| Error Analysis | Misclassified reviews with filters, failure patterns and the robustness set |
| Insights & Analytics | Sentiment shares, net sentiment and trends |

Models and datasets are cached; opening a page never retrains.

## Installation and usage
Requires Python 3.10 or newer. Windows PowerShell commands:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:SENTIMENTIQ_AMAZON_PATH = "C:\Users\HP\Downloads\amazon_fine_food_reviews_cleaned.csv"   # optional
streamlit run app/streamlit_app.py
```

On macOS or Linux, activate with `source .venv/bin/activate` and set the variable with `export`.

**Retrain from scratch** (trained models are included, so this is optional):

```powershell
python scripts/prepare_three_class.py      # build splits and the leakage audit
python scripts/train_models.py             # all six models (add --models naive_bayes logistic_regression linear_svm to skip PyTorch)
python scripts/evaluate.py                 # score every model on the test split once
python scripts/robustness_test.py
python scripts/cross_domain.py --train-cap 6000 --test-cap 3000 --epochs 5
python scripts/update_readme_results.py   # refresh the tables in this README
```

Useful options: `--sample product_food=N` (rows kept per dataset), `--epochs`, `--batch-size`, `--lr`, `--patience`, `--no-tune` (skip the grid search).

## Testing
```powershell
pytest tests -q
```
The 35 tests cover preprocessing and negation handling, label mapping, network output shapes, loading and predicting with all six models, 3x3 confusion matrices in the saved reports, leakage safeguards including near-duplicate detection, local-dataset handling (missing file, chunked sampling, file left unmodified), the two-dataset scope, the cross-domain and pooled experiments, the PyTorch-unavailable path and the business-insight functions.

## Troubleshooting PyTorch on Windows
If `import torch` fails with `WinError 126` (often `shm.dll`):
1. Install the Microsoft Visual C++ Redistributable (x64): https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Or reinstall the CPU build: `pip uninstall -y torch` then `pip install torch --index-url https://download.pytorch.org/whl/cpu`
3. Use 64-bit Python 3.10-3.13 and a short project path.

The app keeps working with the classical models in the meantime.

## Limitations
- Labels are weak, rating-derived; Neutral is rare and noisy, so macro-F1 sits around 0.65.
- Both datasets are English product-style reviews (food and clothing); the models are not expected to work on other languages or very different kinds of text.
- Amazon results use a 50,000-row sample, not the full file.
- Single split and single seed; no confidence intervals.
- Neural networks were trained on CPU with modest budgets and no hyper-parameter search.
- Cross-domain runs used capped data and 5 epochs, so the networks may be under-trained there.
- The bundled datasets have no dates or locations, so trend and per-location insights are tested only on synthetic data.

## Future work
Cross-validation with confidence intervals, tuned or pretrained models (a small transformer), human-labelled Neutral examples, probability calibration, a run on the full Amazon file, and a pilot on a real export with dates and locations.

## Project structure
```
app/          Streamlit app: streamlit_app.py, views/ (one file per page), theme, shared cached loaders
src/          config, settings, datasets, three_class, preprocessing, text_tokens, models, training,
              inference, predict, metrics, cross_domain, insights, viz, compat, nn/ (architectures, trainer)
scripts/      prepare_three_class, train_models, evaluate, robustness_test, cross_domain, update_readme_results
tests/        pytest suite
tests_data/   hand-written robustness reviews
data/processed/   womens_clothing_reviews_cleaned.csv (the Amazon file is read from your own disk)
models/three_class/   trained models and model_card.json
reports/three_class/  dataset audit, evaluation, predictions, cross-domain and robustness reports
```
