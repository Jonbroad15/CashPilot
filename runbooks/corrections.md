# Sync corrections ("sync corrections")

User edited categories/owners in the Sheet's Transactions tab.

1. `python src/sheets.py pull-corrections` → updates `master.csv`, lists changes in `work/corrections.csv`.
2. For each changed category, add/update that merchant in `merchants.csv` (same rules as `categorize.md` #7) so it is right next time. Owner changes → `owner_override` only if it applies to that merchant always.
3. `python src/sheets.py push`.
