import os
import pandas as pd
import pytest

from src.common import MASTER_COLS, drop_known, load_merchants, match_merchant, norm_desc
from src.ingest import account_for, normalize
from src.commit import commit
from src.sheets import build_tabs
from src.common import load_yaml


def w(tmp, name, text):
    p = tmp / name
    p.write_text(text)
    return str(p)


def test_normalize_amex_positive_is_charge(tmp_path):
    p = w(tmp_path, "a.csv", "Date,Description,Amount\n06/01/2025,SOBEYS #934,25.50\n06/02/2025,PAYMENT THANK YOU,-100\n")
    df = normalize(p, "debit")
    assert df.loc[0, "debit"] == 25.5 and pd.isna(df.loc[0, "credit"])
    assert df.loc[1, "credit"] == 100


def test_normalize_bank_positive_is_credit(tmp_path):
    p = w(tmp_path, "a.csv", "date,description,amount\n2025-06-01,PAYROLL,2000\n2025-06-02,RENT,-1500\n")
    df = normalize(p, "credit")
    assert df.loc[0, "credit"] == 2000 and df.loc[1, "debit"] == 1500


def test_normalize_headerless_cibc(tmp_path):
    p = w(tmp_path, "c.csv", '2025-06-01,"SHOPPERS, TORONTO",22.01,\n2025-06-02,PAY,,500\n')
    df = normalize(p)
    assert df.loc[0, "debit"] == 22.01 and df.loc[1, "credit"] == 500


def test_account_for_longest_prefix():
    accts = {"amex": {}, "amex_cobalt": {}}
    assert account_for("amex_cobalt_oct.csv", accts) == "amex_cobalt"
    with pytest.raises(ValueError):
        account_for("mystery.csv", accts)


def test_drop_known_multiset():
    cols = ["date", "description", "debit", "credit", "account"]
    master = pd.DataFrame([["2025-01-01", "X", 5, None, "a"]], columns=cols)
    new = pd.DataFrame([["2025-01-01", "X", 5, None, "a"]] * 2 + [["2025-01-02", "Y", 1, None, "a"]], columns=cols)
    assert len(drop_known(new, master)) == 2  # one X (second same-day purchase) + Y


def test_merchant_match_longest():
    m = pd.DataFrame({"pattern": ["SHOPPERS", "SHOPPERS DRUG MART"], "category": ["a", "b"], "owner_override": ["", "Joint"]})
    m["pattern"] = m["pattern"].map(norm_desc)
    assert match_merchant("Shoppers Drug Mart #13", m) == ("b", "Joint")
    assert match_merchant("nothing", m) == ("", "")


def test_seed_merchants_load():
    assert len(load_merchants()) > 50


def _master(tmp_path):
    p = str(tmp_path / "master.csv")
    pd.DataFrame([["2025-01-01", "OLD", 1.0, None, "groceries", "a", "Jonathan"]], columns=MASTER_COLS).to_csv(p, index=False)
    return p


def test_commit_appends_idempotent_and_archives(tmp_path):
    master, inbox = _master(tmp_path), tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "f.csv").write_text("x")
    new = tmp_path / "new.csv"
    pd.DataFrame([["2025-01-02", "NEW", 2.0, None, "restaurants", "a", "Deweena", "", "f.csv"]],
                 columns=MASTER_COLS + ["needs_review", "source_file"]).to_csv(new, index=False)
    args = (str(new), master, str(inbox), str(tmp_path / "archive"))
    assert commit(*args, archive_files=True) == 1
    assert commit(*args, archive_files=False) == 0
    out = pd.read_csv(master)
    assert len(out) == 2 and out.iloc[-1].owner == "Deweena"
    assert not (inbox / "f.csv").exists()


def test_commit_rejects_blank_category(tmp_path):
    new = tmp_path / "new.csv"
    pd.DataFrame([["2025-01-02", "NEW", 2.0, None, "", "a", "Deweena", "", "f.csv"]],
                 columns=MASTER_COLS + ["needs_review", "source_file"]).to_csv(new, index=False)
    with pytest.raises(SystemExit):
        commit(str(new), _master(tmp_path), str(tmp_path), str(tmp_path / "ar"))


def test_sheet_tabs_shape():
    m = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "master.csv"))
    tabs, months = build_tabs(m, load_yaml("config.yaml"))
    assert len(tabs["Transactions"]) == len(m) + 1
    assert {"Summary", "Monthly", "Transactions", "Categories", "Notes"} <= set(tabs)
    assert set(months) <= set(tabs)  # one tab per month
    assert months[0] == "2023-06"
    assert set(m["owner"]) <= {"Jonathan", "Deweena", "Joint"}
