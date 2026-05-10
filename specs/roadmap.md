# Roadmap

## Shipped

### Data Ingestion
- **Common data model** — `DataCleaner` normalizes any raw bank/credit-card CSV into the standard six-column schema (`date, description, debit, credit, category, account`) via a Claude CLI prompt. Handles Amex sign-convention quirks automatically.
- **Multi-account support** — accounts: `cibc_dividend`, `amex_aeroplan`, `amex_cobalt`, `amex_platinum`, `wealthsimple`, `splitwise`. CSV filename determines account name.
- **Deduplication** — master ledger deduplicates on `(date, description, debit, credit)` so re-processing a file is safe.
- **Automated archiving** — processed CSVs are moved to `data/<month>/` after each pipeline run.

### Classification
- **Sentence-transformer embeddings** — descriptions encoded locally with `all-MiniLM-L6-v2` (384-dim). No external API dependency at inference time.
- **SQLite embedding cache** — `EmbeddingCache` caches embeddings by `SHA-256("<model>:<text>")`, making repeated pipeline runs fast. Model-name prefix in the key prevents stale OpenAI embeddings from being reused after a model switch.
- **CatBoost classifier** — multi-class classifier trained on embeddings + debit/credit amounts; 22 spending categories.
- **Model persistence** — trained models saved to `models/` with UTC timestamps.
- **Retraining script** — `src/retrain.py` retrains from scratch on `master.csv`; required after switching embedding models.

### Analysis & Visualization
- **Category × month pivot** — `analyse()` produces a spending/income summary table printed to stdout after each pipeline run.
- **Bar chart visualization** — `src/plotting.py` renders a seaborn bar chart of spending by category × month from `master.csv`.
- **Cross-validation evaluation** — `src/evaluate.py` runs 5-fold stratified CV, prints a classification report, and optionally saves a confusion-matrix heatmap.

---

## Planned

### Phase 1 — Automated Account Connectivity
Replace manual CSV exports with direct account integrations.

- **Amex integration** — pull transactions from the Amex Canada API or web scraper on a schedule; ingest automatically without user-provided CSVs.
- **Wealthsimple integration** — connect to the Wealthsimple API to pull brokerage and cash account transactions.
- **Scheduled ingestion** — run the pipeline nightly (or on-demand) so the ledger stays current without user action.

### Phase 2 — Local Web Application
Deploy a locally hosted web UI so the user can interact with their finances through a browser rather than the command line.

- **Web framework** — serve the application over a local HTTP server (e.g. FastAPI + a lightweight frontend or a Python-native framework like Streamlit/Dash for rapid iteration).
- **Dashboard** — overview page showing current month's spending by category, progress against targets, and savings goal status at a glance.
- **Transaction browser** — searchable, filterable table of all transactions with the ability to correct a predicted category inline.
- **Category management** — UI for adding, renaming, or merging spending categories without editing CSV files or code.
- **Savings goals UI** — create and track goals; visualize projected completion dates based on current savings rate.
- **Charts & reports** — interactive versions of the existing matplotlib/seaborn plots; filter by date range, account, or category.
- **Pipeline trigger** — button to kick off a new ingestion run (once automated account connectivity is in place) and see live progress.

### Phase 3 — Locally Hosted Web Service (MCP)
Expose the data and pipeline over a local MCP (Model Context Protocol) server so an AI agent can query and act on it.

- **MCP server** — serve `master.csv` data as a queryable resource: transactions, category summaries, monthly totals.
- **Tool endpoints** — expose tools for: fetching spending by category/period, adding/correcting transactions, triggering a pipeline run, querying model predictions.
- **Agent accessibility** — the MCP server runs locally and can be connected to any Claude agent session, enabling natural-language queries like "how much did I spend on restaurants last month?".

### Phase 4 — Agentic Recommendations
Move from passive reporting to proactive, personalized financial guidance.

- **Spending baseline** — compute per-category averages and standard deviations from historical data to establish a personal baseline.
- **Target suggestions** — recommend monthly targets per category based on the user's own history, not generic benchmarks.
- **Overspending alerts** — flag categories where the current month's run-rate is on track to exceed the target; surface mid-month so the user can course-correct.
- **Narrative summaries** — generate human-readable monthly summaries ("You spent 40% more on restaurants than your 3-month average, mostly at weekends.").
- **Trend analysis** — detect multi-month trends (consistently rising utilities, declining grocery spend, etc.) and explain what is driving them.
- **What-if modeling** — allow the agent to answer questions like "if I cut coffee by half, how long until I hit my savings goal?".
