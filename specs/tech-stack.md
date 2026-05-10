# Tech Stack

## Language

- **Python 3** — primary language for all pipeline, ML, and tooling code.

## ML & Data

| Package | Version pin | Role |
|---|---|---|
| `catboost` | latest | Multi-class spending classifier; gradient-boosted trees trained on sentence-transformer embeddings + debit/credit amounts. |
| `sentence-transformers` | latest | Local embedding model (`all-MiniLM-L6-v2`, 384-dim) used to encode transaction descriptions into feature vectors. No API key required. |
| `scikit-learn` | latest | `StratifiedKFold` for cross-validation, `classification_report`, `confusion_matrix` for evaluation metrics. |
| `numpy` | latest | Array operations; feature matrix construction (hstack of embeddings + numeric columns). |
| `pandas` | latest | DataFrame I/O, deduplication, date parsing, pivot-table summaries. |

## Visualization

| Package | Role |
|---|---|
| `matplotlib` | Figure/axes backend for all plots. |
| `seaborn` | Bar charts (spending by category × month), confusion-matrix heatmaps. |

## Storage

| Store | Format | Role |
|---|---|---|
| `master.csv` | CSV | Growing transaction ledger; schema: `date, description, debit, credit, category, account`. Source of truth for analysis and retraining. |
| `embedding_cache.db` | SQLite (stdlib `sqlite3`) | Caches sentence-transformer embeddings keyed by `SHA-256("<model>:<text>")`. Avoids re-encoding repeated descriptions across pipeline runs. |
| `models/*.pkl` | CatBoost pickle | Persisted classifier snapshots; named with UTC timestamp. Incompatible across embedding model versions. |

## LLM Integration

- **Claude CLI** (`claude -p <prompt>`) — invoked via `subprocess.run` inside `DataCleaner` to normalize raw bank CSV files into the standard schema. Uses the locally authenticated Claude Code session; no separate API key is needed.

## Testing

- **pytest** — unit and integration tests covering `DataCleaner` (mocked subprocess), `EmbeddingCache` (in-memory SQLite), and `SpendingClassifier` (small synthetic dataset).

## Notebooks

- **Jupyter** — exploratory analysis and ad-hoc queries on `master.csv`.
