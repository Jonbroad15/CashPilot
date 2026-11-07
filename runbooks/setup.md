# One-time setup

1. `pip install -r requirements.txt`
2. Google Cloud console → create a project → enable **Google Sheets API** → Credentials → *OAuth client ID* → type **Desktop app** → download JSON → save as `~/.config/gspread/credentials.json`.
3. `python src/sheets.py auth` (browser login as the Google account that owns the Sheet; token is cached).
4. Create a blank Google Sheet, copy the ID from its URL into `sheet_id` in `config.yaml`. Share it with your fiancée.
5. Edit `accounts.yaml`: add each of Deweena's and the joint accounts (`owner`, and `positive_is: credit` for bank accounts whose statements show deposits as positive).
6. Drop a first statement in `inbox/` and say "run the budget".
