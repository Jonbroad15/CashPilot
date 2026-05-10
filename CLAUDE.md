# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

CashPilot classifies personal finance transactions using local sentence-transformer embeddings + a CatBoost ML model. Raw CSVs from various bank/credit card accounts are normalized via the Claude CLI, classified into spending categories, and appended to a growing master ledger (`master.csv`).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full labeling pipeline (clean → classify → append to master.csv → archive)
python src/label.py

# Explicit args
python src/label.py --data data --master_csv master.csv --model models/<model>.pkl

# Retrain the classifier from scratch on master.csv (required after switching embedding models)
python src/retrain.py

# Run tests
pytest tests/

# Run a single test file
pytest tests/test_data_cleaner.py -v

# Spending visualization (bar chart by category × month)
python src/plotting.py --master master.csv

# Exploratory notebooks
jupyter notebook notebooks/
```

## Architecture

### Source Files

- **`src/model.py`** — Core ML layer:
  - `DataCleaner`: Calls `claude -p <prompt>` via subprocess to normalize raw CSV into standard columns. No API key required — uses the locally authenticated Claude Code session.
  - `EmbeddingCache`: SQLite-backed cache (`embedding_cache.db`) keyed by SHA-256 of `"<model_name>:<text>"`. The model name prefix ensures cached OpenAI embeddings are never reused after switching models.
  - `get_embeddings()`: Batched embedding generation using `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim), with cache lookup.
  - `SpendingClassifier`: CatBoost classifier; features = sentence-transformer embeddings of description + debit/credit amounts.
  - `EMBEDDING_MODEL`: Module-level constant — change this to swap embedding models (requires retraining).

- **`src/label.py`** — Orchestration:
  - `label()`: Main pipeline — reads `data/`, cleans, classifies, dedupes, appends to `master.csv`
  - `analyse()`: Generates category × month pivot table summaries
  - `clean_data()`: Archives processed CSVs to monthly subdirectories (`data/april/`, etc.)

- **`src/retrain.py`** — Standalone script to retrain `SpendingClassifier` on `master.csv`. Run this whenever the embedding model changes.

- **`src/plotting.py`** — Seaborn/matplotlib bar charts from `master.csv`

### Tests

```
tests/
├── test_data_cleaner.py      # mocks subprocess.run, no real Claude calls
├── test_embeddings.py        # uses real sentence-transformers + in-memory cache
└── test_spending_classifier.py  # trains on a small synthetic dataset
```

### Data

- **`master.csv`**: The growing transaction ledger. Schema: `date, description, debit, credit, category, account`
- **`embedding_cache.db`**: SQLite cache; avoids re-encoding repeated descriptions. Old OpenAI-keyed entries are automatically ignored (different hash prefix).
- **`models/`**: Persisted CatBoost `.pkl` files (named with timestamp). Models trained on OpenAI embeddings are incompatible with the current sentence-transformer features — delete them and retrain.
- **`data/`**: Drop raw bank/credit card CSVs here before running `label.py`; archived after processing

### Accounts
`cibc_dividend`, `amex_aeroplan`, `amex_cobalt`, `amex_platinum`, `wealthsimple`, `splitwise`

### Categories
`housing`, `utilities`, `transit`, `groceries`, `savings`, `entertainment`, `wellness`, `gifts`, `restaurants`, `transfers`, `fees`, `supplies`, `clothing`, `purchases`, `investments`, `income`, `travel`, `rebates`, `cash`, `capital_gl`, `alcohol`, `coffee`

## Data Flow

1. Drop raw CSVs into `data/`
2. `DataCleaner` → `claude -p` normalizes each file into standard schema
3. `SpendingClassifier` → embeds descriptions via `sentence-transformers`, predicts category
4. Results appended to `master.csv`, deduplicated on (date, description, debit, credit), sorted by date
5. Original CSVs archived to `data/<month>/`
