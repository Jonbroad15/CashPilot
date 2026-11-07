"""Step 1: read every statement in inbox/, normalize, drop known rows, pre-fill categories
from merchants.csv, write work/new.csv for the agent to finish categorizing.

Usage: python src/ingest.py [--inbox inbox] [--master master.csv]
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import (MASTER_COLS, clean_money, drop_known, load_merchants, load_yaml,
                        match_merchant, path)

DESC_NAMES = ["description", "merchant", "details", "payee", "name", "memo"]


def account_for(filename, accounts):
    stem = os.path.splitext(filename)[0].lower()
    hits = [a for a in accounts if stem.startswith(a.lower())]
    if not hits:
        raise ValueError(f"'{filename}': filename must start with an account key from accounts.yaml "
                         f"({', '.join(accounts)}). Rename it or add the account.")
    return max(hits, key=len)


def _is_date(x):
    try:
        pd.to_datetime(str(x))
        return not str(x).strip().replace(".", "").isdigit()
    except Exception:
        return False


def normalize(raw_path, positive_is="debit"):
    """Return DataFrame[date, description, debit, credit] from a bank/card export."""
    df = pd.read_csv(raw_path, header=None, dtype=str, keep_default_na=False)
    if df.empty:
        return pd.DataFrame(columns=["date", "description", "debit", "credit", "member"])
    if _is_date(df.iloc[0, 0]):  # header-less (CIBC): date, description, debit, credit
        df = df.iloc[:, :4]
        df.columns = ["date", "description", "debit", "credit"][: df.shape[1]]
    else:
        df.columns = [str(c).strip().lower().replace("_", " ") for c in df.iloc[0]]
        df = df.iloc[1:].reset_index(drop=True)
        cols = list(df.columns)
        date_c = next((c for c in cols if c == "date"), None) or \
            next((c for c in cols if c == "transaction date"), None) or \
            next((c for c in cols if "date" in c), None)
        desc_c = next((c for n in DESC_NAMES for c in cols if c == n), None) or \
            next((c for n in DESC_NAMES for c in cols if n in c), None)
        if not date_c or not desc_c:
            raise ValueError(f"{raw_path}: cannot find date/description columns in {cols}")
        if "transaction type" in cols:  # Wealthsimple: build description from type + payee/merchant + notes
            def nz(c):
                return df[c].str.strip() if c in cols else ""
            ttype = nz("transaction type").where(~nz("transaction type").isin(["Purchase", "Spend"]), "")
            parts = [ttype, nz("transaction sub type"), nz("payee"), nz("merchant"), nz("notes")]
            desc = pd.Series("", index=df.index)
            for p_ in parts:
                desc = desc + " " + p_
            df["description"] = desc
            desc_c = "description"
            if "status" in cols:  # pending / failed / cancelled rows are not real transactions
                df = df[df["status"].str.strip().str.lower() == "completed"].reset_index(drop=True)
        out = pd.DataFrame({"date": df[date_c], "description": df[desc_c]})
        mem_c = next((c for c in cols if c in ("card member", "cardholder", "card holder")), None)
        out["member"] = df[mem_c] if mem_c else ""
        deb_c = next((c for c in cols if c in ("debit", "withdrawal", "withdrawals")), None)
        cre_c = next((c for c in cols if c in ("credit", "deposit", "deposits")), None)
        if deb_c or cre_c:
            out["debit"] = df[deb_c] if deb_c else ""
            out["credit"] = df[cre_c] if cre_c else ""
        else:
            amt_c = next((c for c in cols if c in ("amount", "transaction amount")), None) or \
                next((c for c in cols if "amount" in c), None)
            if not amt_c:
                raise ValueError(f"{raw_path}: cannot find amount columns in {cols}")
            amt = pd.to_numeric(df[amt_c].str.replace(r"[$,\s]", "", regex=True), errors="coerce")
            pos, neg = amt.where(amt > 0), (-amt).where(amt < 0)
            out["debit"], out["credit"] = (pos, neg) if positive_is == "debit" else (neg, pos)
        df = out
    for c in ("debit", "credit"):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(r"[$,\s]", "", regex=True), errors="coerce")
    if "member" not in df:
        df["member"] = ""
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"])
    df = df[df["debit"].notna() | df["credit"].notna()]
    df["description"] = df["description"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return df[["date", "description", "debit", "credit", "member"]]


def ingest(inbox, master_csv, out_csv):
    accounts = load_yaml("accounts.yaml")
    merchants = load_merchants()
    master = pd.read_csv(master_csv) if os.path.exists(master_csv) else pd.DataFrame(columns=MASTER_COLS)
    files = sorted(f for f in os.listdir(inbox) if f.lower().endswith(".csv") and not f.startswith("."))
    frames = []
    for f in files:
        acct = account_for(f, accounts)
        info = accounts[acct]
        df = normalize(os.path.join(inbox, f), info.get("positive_is", "debit"))
        members = {k.upper(): v for k, v in (info.get("card_members") or {}).items()}

        def owner_of(m, default=info["owner"]):
            return next((o for k, o in members.items() if k in str(m).upper()), default)

        df["account"], df["source_file"] = acct, f
        df["owner"] = [owner_of(m) for m in df["member"]]
        frames.append(df)
        print(f"{f}: {len(df)} rows -> {acct} ({info['owner']})")
    if not frames:
        print("inbox is empty - nothing to do")
        return pd.DataFrame()
    new = pd.concat(frames, ignore_index=True)
    before = len(new)
    alias = {a: k for k, v in accounts.items() for a in (v.get("aliases") or [])}
    known = master.assign(account=master["account"].replace(alias))
    new = drop_known(new, known).reset_index(drop=True)
    print(f"{before - len(new)} rows already in master skipped, {len(new)} new")
    cats = [match_merchant(d, merchants) for d in new["description"]]
    new["category"] = [c for c, _ in cats]
    over = [o for _, o in cats]
    new["owner"] = [o or cur for o, cur in zip(over, new["owner"])]
    new["needs_review"] = ""
    new = new[MASTER_COLS + ["needs_review", "source_file"]]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    new.to_csv(out_csv, index=False)
    blank = (new["category"] == "").sum()
    print(f"{len(new) - blank} categorized from merchants.csv, {blank} need the agent -> {out_csv}")
    return new


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", default=path("inbox"))
    ap.add_argument("--master", default=path("master.csv"))
    ap.add_argument("--out", default=path("work", "new.csv"))
    a = ap.parse_args()
    ingest(a.inbox, a.master, a.out)
