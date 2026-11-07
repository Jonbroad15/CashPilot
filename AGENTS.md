# Household Budget (Jonathan & Deweena)

Drop bank/credit-card statements in `inbox/`, say **"run the budget"**, and follow `runbooks/run.md`. Output is a Google Sheet (Summary, Monthly, Transactions, Categories, Notes) plus the archive `master.csv`.

| File | Purpose |
|---|---|
| `runbooks/run.md` | Main procedure (start here) |
| `runbooks/categorize.md` | Category definitions, examples, owner rules |
| `runbooks/corrections.md` | "sync corrections" after edits in the Sheet |
| `runbooks/setup.md` | One-time Google auth + Sheet setup |
| `accounts.yaml` | statement filename prefix → owner (Deweena / Joint / Jonathan) |
| `config.yaml` | Sheet ID, categories and their type (spending/income/excluded) |
| `merchants.csv` | merchant → category memory; grow it |
| `master.csv` | Archive: `date,description,debit,credit,category,account,owner` (never overwrite; commit.py appends) |
| `src/ingest.py` `commit.py` `sheets.py` | Deterministic helpers; you do the judgment |
| `legacy/` | Old CatBoost pipeline, unused |

Tests: `pytest tests/`. Statement data is gitignored—never commit it.
