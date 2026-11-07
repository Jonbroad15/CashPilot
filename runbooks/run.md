# Run the budget

Trigger: user says "run the budget" (statements are in `inbox/`). Work from the repo root. Be hands-off: only ask the user if an account is unknown.

1. **Sync edits first** (skip if `sheet_id` is empty): `python src/sheets.py pull-corrections`. If it changed rows, update `merchants.csv` from `work/corrections.csv` (see `corrections.md`).
2. **Ingest**: `python src/ingest.py`
   - Unknown filename prefix → ask the user which account/owner it is, add it to `accounts.yaml`, retry.
   - Not a CSV (PDF/XLSX)? Convert it to CSV in `inbox/` yourself first, keeping the filename prefix. If columns can't be parsed, hand-write `work/new.csv` with the same columns.
   - Amount column sign wrong (e.g. refunds look like spending)? Set `positive_is` for that account in `accounts.yaml`.
3. **Categorize**: follow `runbooks/categorize.md` on `work/new.csv`.
4. **Commit**: `python src/commit.py` (validates, backs up, appends to `master.csv`, moves statements to `archive/`).
5. **Push**: `python src/sheets.py push`
6. **Report** in ≤8 lines: files/rows added per owner, spend per owner for the latest month, rows flagged `?` for review (show them), and the Sheet link.

Rules: never delete raw statements or `master.csv`; re-running must be safe (duplicates are skipped).
