"""Google Sheets I/O. Raw data are values; every derived number is a live formula.

  python src/sheets.py auth               one-time browser login
  python src/sheets.py push               rewrite all tabs from master.csv
  python src/sheets.py pull-corrections   copy category/owner edits made in the Sheet into master.csv
  python src/sheets.py push --dry-run     write tab previews to work/sheet_preview/ (no network)
"""
import argparse
import datetime
import os
import shutil
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import KEY_COLS, OWNERS, key_frame, load_yaml, path

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
TX_HEADER = ["date", "description", "debit", "credit", "category", "account", "owner",
             "month", "amount", "type"]


def _num(x):
    return "" if pd.isna(x) else float(x)


def _s(x):
    return "" if pd.isna(x) else str(x)


def transactions_tab(master):
    rows = [TX_HEADER]
    for i, r in enumerate(master.itertuples(index=False), start=2):
        rows.append([_s(r.date), _s(r.description), _num(r.debit), _num(r.credit), _s(r.category),
                     _s(r.account), _s(r.owner), f"=YEAR(A{i})*100+MONTH(A{i})", f"=N(D{i})-N(C{i})",
                     f'=IFERROR(VLOOKUP(E{i},Categories!$A:$B,2,FALSE),"unknown")'])
    return rows


def categories_tab(cfg):
    rows = [["category", "type", "description (type: spending | income | excluded; Monthly tab follows edits live, ask the agent to re-push to rebuild Summary rows)"]]
    rows += [[k, v["type"], v["desc"]] for k, v in cfg["categories"].items()]
    return rows


def _section(title, cats, sign, start):
    """Category x owner block. Returns (rows, next_row, total_row_number)."""
    rows = [[title], ["category"] + OWNERS + ["Total"]]
    hdr = start + 1
    for j, c in enumerate(cats):
        r = start + 2 + j
        row = [c]
        for col in "BCD":
            row.append(f"={sign}SUMIFS(Transactions!$I:$I,Transactions!$E:$E,$A{r},"
                       f"Transactions!$G:$G,{col}${hdr},Transactions!$H:$H,$B$3)")
        row.append(f"=SUM(B{r}:D{r})")
        rows.append(row)
    first, last = start + 2, start + 1 + len(cats)
    t = last + 1
    rows.append(["TOTAL"] + [f"=SUM({c}{first}:{c}{last})" for c in "BCDE"])
    return rows, t + 2, t


def summary_tab(cfg, period="All", title="Household budget summary"):
    spend = [k for k, v in cfg["categories"].items() if v["type"] == "spending"]
    inc = [k for k, v in cfg["categories"].items() if v["type"] == "income"]
    label = ("Period (pick a month as yyyymm e.g. 202506, or All)" if period == "All"
             else "Month (yyyymm)")
    rows = [[title],
            [label, period],
            ["criteria used (helper)", '=IF(B2="All",">0",B2)'],
            [""]]
    s_rows, nxt, s_tot = _section("SPENDING by category and owner (refunds reduce spending)", spend, "-", 5)
    rows += s_rows + [[""]]
    i_rows, nxt, i_tot = _section("INCOME by category and owner", inc, "", nxt)
    rows += i_rows + [[""]]
    rows.append(["NET (income minus spending)"] + [f"={c}{i_tot}-{c}{s_tot}" for c in "BCDE"])
    return rows


def monthly_tab(months):
    hdr = ["month (yyyymm)"] + [f"spending {o}" for o in OWNERS] + ["spending Total"] + \
          [f"income {o}" for o in OWNERS] + ["income Total", "net"]
    rows = [hdr]
    for i, m in enumerate(months, start=2):
        row = [int(m.replace("-", ""))]
        for typ, sign in (("spending", "-"), ("income", "")):
            for o in OWNERS:
                row.append(f'={sign}SUMIFS(Transactions!$I:$I,Transactions!$G:$G,"{o}",'
                           f'Transactions!$H:$H,$A{i},Transactions!$J:$J,"{typ}")')
            c0, c1 = ("B", "D") if typ == "spending" else ("F", "H")
            row.append(f"=SUM({c0}{i}:{c1}{i})")
        row.append(f"=I{i}-E{i}")
        rows.append(row)
    return rows


NOTES = [
    ["How this workbook works"],
    ["Raw data: the Transactions tab columns A-G come straight from card/bank statements (plus our category and owner labels). Everything else is a formula."],
    [""],
    ["Transactions columns"],
    ["month = YEAR(date)*100+MONTH(date), e.g. 202506  |  amount = credit - debit (negative = money out, positive = money in)"],
    ["type = looked up from the Categories tab: spending, income or excluded. Excluded rows (transfers, credit-card payments) are ignored in every total, so money moved between our own accounts is not double counted."],
    [""],
    ["Summary tab"],
    ["Pick a period in B2 (yyyymm such as 202506, or All). Each month also has its own tab (named yyyy-mm) with the same layout. Each cell = SUMIFS over Transactions for that category, owner and month. B3 turns 'All' into the criterion >0 (any month)."],
    ["Spending cells are shown positive: -SUMIFS(amount). A refund (credit) in a category reduces that category's spending."],
    ["Income cells = SUMIFS(amount). NET = total income - total spending, per owner and overall."],
    ["Owners: Deweena, Joint and Jonathan come from accounts.yaml (default per account) or a merchant override. Total = sum of the three."],
    [""],
    ["Monthly tab"],
    ["One row per month: spending and income per owner using the type column; net = income total - spending total."],
    [""],
    ["Editing"],
    ["Fix a wrong category/owner directly in Transactions columns E/G, then ask the agent to 'sync corrections' so master.csv and merchants.csv learn it. Changing a category type on the Categories tab updates the Monthly tab live; re-push to rebuild Summary."],
    ["Do not type over formula columns H-J. Do not sort/insert rows in Transactions without asking the agent to re-push afterwards."],
    [f"Last pushed: {datetime.datetime.now():%Y-%m-%d %H:%M}"],
]


def build_tabs(master, cfg):
    master = master.sort_values("date", kind="stable").reset_index(drop=True)
    months = pd.period_range(pd.to_datetime(master["date"]).min(), pd.to_datetime(master["date"]).max(), freq="M")
    months = [str(m) for m in months]
    month_tabs = {f"{m}": summary_tab(cfg, int(m.replace("-", "")), f"Spending and income for {m}")
                  for m in months[::-1]}
    return {"Summary": summary_tab(cfg), "Monthly": monthly_tab(months), **month_tabs,
            "Transactions": transactions_tab(master), "Categories": categories_tab(cfg),
            "Notes": NOTES}, months


def _client():
    import gspread
    return gspread.oauth(scopes=SCOPES)


def _open(cfg):
    if not cfg.get("sheet_id"):
        raise SystemExit("config.yaml: sheet_id is empty (see runbooks/setup.md)")
    return _client().open_by_key(cfg["sheet_id"])


def push(master_csv, dry_run=False):
    cfg = load_yaml("config.yaml")
    master = pd.read_csv(master_csv)
    # master.csv is the archive; the Sheet always mirrors it
    master = master.sort_values("date", kind="stable").reset_index(drop=True)
    master.to_csv(master_csv, index=False)
    tabs, months = build_tabs(master, cfg)
    if dry_run:
        out = path("work", "sheet_preview")
        os.makedirs(out, exist_ok=True)
        for name, rows in tabs.items():
            pd.DataFrame(rows).to_csv(os.path.join(out, f"{name}.csv"), index=False, header=False)
        print(f"preview written to {out}")
        return tabs
    ss = _open(cfg)
    existing = {w.title: w for w in ss.worksheets()}
    # 1) create/resize/order all tabs in a single request (Sheets write quota is ~60/min)
    reqs = []
    for idx, (name, rows) in enumerate(tabs.items()):
        grid = {"rowCount": len(rows) + 50, "columnCount": max(len(r) for r in rows) + 2,
                "frozenRowCount": 1 if name in ("Transactions", "Monthly", "Categories") else 0}
        if name in existing:
            reqs.append({"updateSheetProperties": {"properties": {
                "sheetId": existing[name].id, "index": idx, "gridProperties": grid},
                "fields": "index,gridProperties(rowCount,columnCount,frozenRowCount)"}})
        else:
            reqs.append({"addSheet": {"properties": {"title": name, "index": idx, "gridProperties": grid}}})
    ss.batch_update({"requests": reqs})
    # 2) clear + write every tab in two calls
    ss.client.request("post", f"https://sheets.googleapis.com/v4/spreadsheets/{ss.id}/values:batchClear",
                      json={"ranges": [f"'{n}'" for n in tabs]})
    ss.values_batch_update({"valueInputOption": "USER_ENTERED", "data": [
        {"range": f"'{n}'!A1", "values": r} for n, r in tabs.items()]})
    for n, r in tabs.items():
        print(f"  {n}: {len(r)} rows")
    ws_ids = {w.title: w.id for w in ss.worksheets()}
    dv = {"setDataValidation": {
        "range": {"sheetId": ws_ids["Summary"], "startRowIndex": 1, "endRowIndex": 2,
                  "startColumnIndex": 1, "endColumnIndex": 2},
        "rule": {"condition": {"type": "ONE_OF_LIST", "values": [{"userEnteredValue": v} for v in
                                                                 ["All"] + [m.replace("-", "") for m in months[::-1]]]},
                 "showCustomUi": True, "strict": True}}}
    ss.batch_update({"requests": [dv] + format_requests(ws_ids, tabs)})
    if "Sheet1" in existing:
        ss.del_worksheet(existing["Sheet1"])
    print(f"pushed -> {ss.url}")


# ---------------------------------------------------------------- formatting
NAVY, BLUE, LBLUE = "#1F3864", "#2F5597", "#DCE6F5"
GREY, LGREY, GREEN, LGREEN, RED, LRED = "#595959", "#F2F2F2", "#2E7D32", "#E2F0D9", "#C0392B", "#FBE4E1"
OWNER_FILL = {"Deweena": "#EADCF4", "Joint": "#FCE8CF", "Jonathan": "#D4ECEC"}
MONEY = "$#,##0.00;[Red]-$#,##0.00;\"-\""


def _rgb(h):
    h = h.lstrip("#")
    return {"red": int(h[0:2], 16) / 255, "green": int(h[2:4], 16) / 255, "blue": int(h[4:6], 16) / 255}


def _fmt(sid, r0, r1, c0, c1, bg=None, fg=None, bold=None, size=None, num=None, align=None, wrap=None):
    f, fields = {}, []
    if bg:
        f["backgroundColor"] = _rgb(bg); fields.append("backgroundColor")
    tf = {}
    if fg:
        tf["foregroundColor"] = _rgb(fg)
    if bold is not None:
        tf["bold"] = bold
    if size:
        tf["fontSize"] = size
    if tf:
        f["textFormat"] = tf; fields.append("textFormat")
    if num:
        f["numberFormat"] = {"type": "NUMBER", "pattern": num}; fields.append("numberFormat")
    if align:
        f["horizontalAlignment"] = align; fields.append("horizontalAlignment")
    if wrap:
        f["wrapStrategy"] = wrap; fields.append("wrapStrategy")
    return {"repeatCell": {"range": {"sheetId": sid, "startRowIndex": r0, "endRowIndex": r1,
                                     "startColumnIndex": c0, "endColumnIndex": c1},
                           "cell": {"userEnteredFormat": f}, "fields": "userEnteredFormat(" + ",".join(fields) + ")"}}


def _width(sid, c0, c1, px):
    return {"updateDimensionProperties": {"range": {"sheetId": sid, "dimension": "COLUMNS",
                                                    "startIndex": c0, "endIndex": c1},
                                          "properties": {"pixelSize": px}, "fields": "pixelSize"}}


def _border(sid, r0, r1, c0, c1, side="bottom", color="#D9D9D9", style="SOLID"):
    return {"updateBorders": {"range": {"sheetId": sid, "startRowIndex": r0, "endRowIndex": r1,
                                        "startColumnIndex": c0, "endColumnIndex": c1},
                              side: {"style": style, "color": _rgb(color)}}}


def _props(sid, color, hide_grid=False):
    return {"updateSheetProperties": {"properties": {"sheetId": sid, "tabColor": _rgb(color),
                                                     "gridProperties": {"hideGridlines": hide_grid}},
                                      "fields": "tabColor,gridProperties.hideGridlines"}}


def format_requests(ids, tabs):
    """Colors, number formats, widths, borders, tab colors. Pure formatting, no data."""
    out = []
    for name, rows in tabs.items():
        sid, n = ids[name], len(rows)
        if name == "Transactions":
            out += [_props(sid, GREEN),
                    _fmt(sid, 0, 1, 0, 10, bg=NAVY, fg="#FFFFFF", bold=True, align="CENTER"),
                    _fmt(sid, 1, n, 0, 1, num="yyyy-mm-dd"),
                    _fmt(sid, 1, n, 2, 4, num="#,##0.00"), _fmt(sid, 1, n, 8, 9, num=MONEY),
                    _fmt(sid, 1, n, 7, 8, num="0", align="CENTER"),
                    _fmt(sid, 1, n, 4, 7, align="LEFT"),
                    _width(sid, 0, 1, 95), _width(sid, 1, 2, 420), _width(sid, 2, 4, 90),
                    _width(sid, 4, 7, 140), _width(sid, 7, 10, 90),
                    {"setBasicFilter": {"filter": {"range": {"sheetId": sid, "startRowIndex": 0, "endRowIndex": n,
                                                             "startColumnIndex": 0, "endColumnIndex": 10}}}}]
        elif name == "Categories":
            out += [_props(sid, "#7F7F7F"), _fmt(sid, 0, 1, 0, 3, bg=NAVY, fg="#FFFFFF", bold=True),
                    _width(sid, 0, 1, 130), _width(sid, 1, 2, 90), _width(sid, 2, 3, 700)]
            for i, r in enumerate(rows[1:], start=1):
                col = {"spending": LRED, "income": LGREEN, "excluded": LGREY}.get(r[1])
                if col:
                    out.append(_fmt(sid, i, i + 1, 1, 2, bg=col, align="CENTER"))
        elif name == "Notes":
            out += [_props(sid, "#7F7F7F", True), _width(sid, 0, 1, 950),
                    _fmt(sid, 0, n, 0, 1, wrap="WRAP"), _fmt(sid, 0, 1, 0, 1, bold=True, size=14, fg=NAVY)]
            for i, r in enumerate(rows):
                if i and r[0] in ("Transactions columns", "Summary tab", "Monthly tab", "Editing"):
                    out.append(_fmt(sid, i, i + 1, 0, 1, bold=True, size=11, bg=LBLUE, fg=NAVY))
        elif name == "Monthly":
            out += [_props(sid, BLUE),
                    _fmt(sid, 0, 1, 0, 10, bg=NAVY, fg="#FFFFFF", bold=True, align="CENTER", wrap="WRAP"),
                    _fmt(sid, 0, 1, 1, 5, bg="#9C2B21"), _fmt(sid, 0, 1, 5, 9, bg="#2E6B30"),
                    _fmt(sid, 1, n, 0, 1, num="0", bold=True, align="CENTER"),
                    _fmt(sid, 1, n, 1, 10, num=MONEY),
                    _fmt(sid, 1, n, 4, 5, bold=True, bg=LRED), _fmt(sid, 1, n, 8, 9, bold=True, bg=LGREEN),
                    _fmt(sid, 1, n, 9, 10, bold=True, bg=LBLUE),
                    _border(sid, 1, n, 0, 10), _width(sid, 0, 1, 110), _width(sid, 1, 10, 115)]
        else:  # Summary and the per-month tabs
            tab_color = BLUE if name == "Summary" else "#8EA9DB"
            out += [_props(sid, tab_color, True), _width(sid, 0, 1, 300), _width(sid, 1, 5, 125),
                    _fmt(sid, 0, 1, 0, 5, bold=True, size=16, fg=NAVY),
                    _fmt(sid, 1, 2, 0, 1, bold=True), _fmt(sid, 1, 2, 1, 2, bold=True, bg="#FFF2CC", num="0", align="CENTER"),
                    _fmt(sid, 2, 3, 0, 3, fg="#A6A6A6", size=9, num="0")]
            sec_color = None
            for i, r in enumerate(rows):
                a = str(r[0])
                if a.startswith("SPENDING"):
                    sec_color = RED
                    out.append(_fmt(sid, i, i + 1, 0, 5, bg=RED, fg="#FFFFFF", bold=True, size=12))
                elif a.startswith("INCOME"):
                    sec_color = GREEN
                    out.append(_fmt(sid, i, i + 1, 0, 5, bg=GREEN, fg="#FFFFFF", bold=True, size=12))
                elif a == "category":
                    out.append(_fmt(sid, i, i + 1, 0, 5, bg=LGREY, bold=True, align="CENTER"))
                    for j, o in enumerate(OWNERS, start=1):
                        out.append(_fmt(sid, i, i + 1, j, j + 1, bg=OWNER_FILL[o]))
                    out.append(_fmt(sid, i, i + 1, 0, 1, align="LEFT"))
                    start = i + 1
                elif a == "TOTAL":
                    out += [_fmt(sid, start, i, 1, 5, num=MONEY), _border(sid, start, i, 0, 5),
                            _fmt(sid, start, i, 4, 5, bold=True),
                            _fmt(sid, i, i + 1, 0, 5, bg=LRED if sec_color == RED else LGREEN, bold=True, num=MONEY),
                            _border(sid, i, i + 1, 0, 5, "top", GREY)]
                elif a.startswith("NET"):
                    out += [_fmt(sid, i, i + 1, 0, 5, bg=LBLUE, bold=True, size=12, num=MONEY),
                            _border(sid, i, i + 1, 0, 5, "top", NAVY, "SOLID_MEDIUM")]
    return out


def pull_corrections(master_csv):
    cfg = load_yaml("config.yaml")
    master = pd.read_csv(master_csv)
    ws = _open(cfg).worksheet("Transactions")
    vals = ws.get_all_values(value_render_option="UNFORMATTED_VALUE")[1:]
    sheet = pd.DataFrame([r[:7] + [""] * (7 - len(r[:7])) for r in vals],
                         columns=["date", "description", "debit", "credit", "category", "account", "owner"])
    sheet = sheet[sheet["description"] != ""]
    # unformatted dates come back as serial numbers
    if pd.api.types.is_numeric_dtype(pd.to_numeric(sheet["date"], errors="coerce")) and \
            pd.to_numeric(sheet["date"], errors="coerce").notna().all():
        sheet["date"] = (pd.Timestamp("1899-12-30") + pd.to_timedelta(sheet["date"].astype(float), unit="D")).dt.strftime("%Y-%m-%d")
    slots = defaultdict(list)
    for idx, k in enumerate(key_frame(master)):
        slots[k].append(idx)
    changes = []
    for k, (_, r) in zip(key_frame(sheet), sheet.iterrows()):
        if not slots[k]:
            continue
        i = slots[k].pop(0)
        for col, ok in (("category", r["category"] in cfg["categories"]), ("owner", r["owner"] in OWNERS)):
            if ok and master.at[i, col] != r[col]:
                changes.append((master.at[i, "description"], col, master.at[i, col], r[col]))
                master.at[i, col] = r[col]
    if changes:
        os.makedirs(path("archive"), exist_ok=True)
        shutil.copy(master_csv, path("archive", f"master.{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"))
        master.to_csv(master_csv, index=False)
        pd.DataFrame(changes, columns=["description", "field", "old", "new"]).to_csv(path("work", "corrections.csv"), index=False)
    print(f"{len(changes)} corrections applied" + (" -> work/corrections.csv" if changes else ""))
    for c in changes[:30]:
        print("  ", c)
    return changes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["auth", "push", "pull-corrections"])
    ap.add_argument("--master", default=path("master.csv"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cmd == "auth":
        _client(); print("authenticated")
    elif a.cmd == "push":
        push(a.master, a.dry_run)
    else:
        pull_corrections(a.master)
