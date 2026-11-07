"""Shared helpers: paths, config, merchant matching, duplicate handling."""
import os
import re
import yaml
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER_COLS = ["date", "description", "debit", "credit", "category", "account", "owner"]
KEY_COLS = ["date", "description", "debit", "credit", "account"]
OWNERS = ["Deweena", "Joint", "Jonathan"]


def path(*p):
    return os.path.join(ROOT, *p)


def load_yaml(name):
    with open(path(name)) as f:
        return yaml.safe_load(f) or {}


def norm_desc(s):
    """Uppercase, strip digits/#/punctuation noise so merchants match across statements."""
    s = re.sub(r"[#\d]+", " ", str(s).upper())
    s = re.sub(r"[^A-Z&' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_merchants(p=None):
    p = p or path("merchants.csv")
    if not os.path.exists(p):
        return pd.DataFrame(columns=["pattern", "category", "owner_override"])
    m = pd.read_csv(p, dtype=str).fillna("")
    m["pattern"] = m["pattern"].map(norm_desc)
    return m[m["pattern"].str.len() >= 4]


def match_merchant(desc, merchants):
    """Return (category, owner_override) for the longest matching pattern, else ('','')."""
    d = norm_desc(desc)
    best = None
    for pat, cat, own in merchants[["pattern", "category", "owner_override"]].itertuples(index=False):
        if pat in d and (best is None or len(pat) > len(best[0])):
            best = (pat, cat, own)
    return (best[1], best[2]) if best else ("", "")


def clean_money(df):
    for c in ("debit", "credit"):
        df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
    return df


def key_frame(df):
    k = df[KEY_COLS].copy()
    k["debit"] = pd.to_numeric(k["debit"], errors="coerce").fillna(0).round(2)
    k["credit"] = pd.to_numeric(k["credit"], errors="coerce").fillna(0).round(2)
    k["date"] = pd.to_datetime(k["date"]).dt.strftime("%Y-%m-%d")
    return k.astype(str).agg("|".join, axis=1)


def drop_known(new_df, master_df):
    """Remove incoming rows already in master. Multiset-aware: identical same-day rows
    inside one statement are kept, overlapping statements are not double counted."""
    if master_df.empty or new_df.empty:
        return new_df
    seen = key_frame(master_df).value_counts().to_dict()
    keep = []
    for k in key_frame(new_df):
        if seen.get(k, 0) > 0:
            seen[k] -= 1
            keep.append(False)
        else:
            keep.append(True)
    return new_df[keep]
