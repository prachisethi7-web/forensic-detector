"""
data_sources.py - everything that talks to the internet.
Every function returns None (or an empty list) on failure instead of raising,
so the app can say "could not fetch" and carry on.

NOTE: written without live network access to Yahoo/NSE; row labels and endpoints
follow current public behaviour and may need small adjustments after the first live run.
"""
import re

import pandas as pd

# our field name -> (statement, [possible Yahoo row labels in order of preference])
FIELD_MAP = {
    "receivables":      ("bs", ["Receivables", "Accounts Receivable", "Gross Accounts Receivable"]),
    "sales":            ("is", ["Total Revenue", "Operating Revenue"]),
    "cogs":             ("is", ["Cost Of Revenue", "Reconciled Cost Of Revenue"]),
    "currentAssets":    ("bs", ["Current Assets", "Total Current Assets"]),
    "ppe":              ("bs", ["Net PPE", "Net Property Plant And Equipment"]),
    "totalAssets":      ("bs", ["Total Assets"]),
    "depreciation":     ("is", ["Reconciled Depreciation", "Depreciation And Amortization In Income Statement"]),
    "sga":              ("is", ["Selling General And Administration", "Selling General And Administrative"]),
    "ltDebt":           ("bs", ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"]),
    "currentLiab":      ("bs", ["Current Liabilities", "Total Current Liabilities"]),
    "netIncome":        ("is", ["Net Income", "Net Income Common Stockholders"]),
    "ocf":              ("cf", ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]),
    "totalLiab":        ("bs", ["Total Liabilities Net Minority Interest", "Total Liabilities"]),
    "retainedEarnings": ("bs", ["Retained Earnings"]),
}


def _pick(df, labels, col):
    if df is None or df.empty:
        return None
    for lab in labels:
        if lab in df.index:
            v = df.loc[lab, col]
            if pd.notna(v):
                return float(v)
    return None


def _num(x):
    try:
        x = float(x)
        return x if x == x and x > 0 else None
    except Exception:
        return None


def get_market_cap(t, balance_sheet=None):
    """Try several routes, because Yahoo's .info often fails on cloud servers. Returns (value, source) or (None, None)."""
    try:
        v = _num((t.info or {}).get("marketCap"))
        if v:
            return v, "Yahoo info"
    except Exception:
        pass
    fi = None
    try:
        fi = t.fast_info
    except Exception:
        pass
    if fi is not None:
        def g(key):
            for acc in (lambda: fi[key], lambda: getattr(fi, key)):
                try:
                    v = _num(acc())
                    if v:
                        return v
                except Exception:
                    pass
            return None
        v = g("market_cap")
        if v:
            return v, "Yahoo fast_info"
        price, shares = g("last_price"), g("shares")
        if price and shares:
            return price * shares, "price x shares (fast_info)"
    # last resort: latest close x share count from the balance sheet
    try:
        px = _num(t.history(period="5d")["Close"].dropna().iloc[-1])
        bs = balance_sheet
        if px and bs is not None:
            for lab in ("Ordinary Shares Number", "Share Issued"):
                if lab in bs.index:
                    sh = _num(bs.loc[lab].dropna().iloc[0])
                    if sh:
                        return px * sh, "latest close x balance-sheet shares"
    except Exception:
        pass
    return None, None


def search_company(query):
    """Name or ticker -> list of {symbol, name, exchange}. NSE (.NS) / BSE (.BO) only."""
    import yfinance as yf
    out, q = [], query.strip()
    if not q:
        return out
    try:
        for r in (yf.Search(q, max_results=10).quotes or []):
            sym = r.get("symbol", "")
            if sym.endswith((".NS", ".BO")):
                out.append({"symbol": sym, "name": r.get("shortname") or r.get("longname") or sym,
                            "exchange": "NSE" if sym.endswith(".NS") else "BSE"})
    except Exception:
        pass
    if not out and re.fullmatch(r"[A-Za-z0-9&-]{1,20}", q):      # looks like a ticker already
        out.append({"symbol": q.upper() + ".NS", "name": q.upper(), "exchange": "NSE (guessed)"})
    return out


def get_financials(symbol):
    """
    Returns {"prior": {...}, "current": {...}, "missing": [...], "periods": (prior_date, current_date)}
    or None. Missing items are None. Market equity uses today's market cap (approximation).
    """
    import yfinance as yf
    try:
        t = yf.Ticker(symbol)
        stm = {"is": t.income_stmt, "bs": t.balance_sheet, "cf": t.cashflow}
        base = stm["is"]
        if base is None or base.shape[1] < 2:
            return None
        cols = sorted(base.columns, reverse=True)[:2]          # latest two annual periods
        cur, pri = {}, {}
        for key, (kind, labels) in FIELD_MAP.items():
            df = stm[kind]
            for target, col in ((cur, cols[0]), (pri, cols[1])):
                target[key] = _pick(df, labels, col) if df is not None and col in df.columns else None
        mc, mc_src = get_market_cap(t, stm["bs"])
        cur["marketEquity"] = mc
        pri["marketEquity"] = None
        missing = sorted({k for k, v in cur.items() if v is None} | {k for k, v in pri.items()
                         if v is None and k not in ("marketEquity", "totalLiab", "retainedEarnings")})
        return {"prior": pri, "current": cur, "missing": missing, "market_cap_source": mc_src,
                "periods": (str(cols[1])[:10], str(cols[0])[:10])}
    except Exception:
        return None


def get_quarterly_net_income(symbol):
    """Yahoo only provides a few recent quarters, so this is usually too short for SARIMA."""
    import yfinance as yf
    try:
        q = yf.Ticker(symbol).quarterly_income_stmt
        if q is None or "Net Income" not in q.index:
            return None
        s = q.loc["Net Income"].dropna().sort_index()
        return s if len(s) else None
    except Exception:
        return None


def get_announcements(symbol):
    """Recent NSE corporate announcements (unofficial endpoint; often blocked from cloud servers)."""
    import requests
    sym = symbol.replace(".NS", "").replace(".BO", "")
    hdr = {"User-Agent": "Mozilla/5.0", "Accept": "application/json", "Accept-Language": "en-US,en;q=0.9"}
    try:
        s = requests.Session()
        s.headers.update(hdr)
        s.get("https://www.nseindia.com", timeout=10)                 # warm cookies
        r = s.get("https://www.nseindia.com/api/corporate-announcements",
                  params={"index": "equities", "symbol": sym}, timeout=15)
        data = r.json()
        return [{"date": str(x.get("an_dt") or x.get("sort_date") or ""),
                 "text": " ".join(str(x.get("desc") or "").split()),
                 "detail": " ".join(str(x.get("attchmntText") or "").split())}
                for x in data][:200]
    except Exception:
        return None


# Announcement subject lines -> flag (the subject line is the verbatim quote)
ANN_RULES = [
    ("auditor_change", r"(resign|cessation|change|appointment)[^.]{0,50}(statutory auditor|auditors?)|auditor[^.]{0,30}resign", "high",
     "The company announced a change involving its statutory auditor."),
    ("restatement", r"restat", "high", "The announcement refers to a restatement of financial results."),
    ("management_turnover", r"(resign|cessation)[^.]{0,50}(chief financial officer|CFO|company secretary|chief accounting)", "low",
     "A finance or compliance officer's departure was announced; the announcement does not state an accounting reason."),
]


def classify_announcements(items):
    flags, seen = [], set()
    for it in items or []:
        txt = it["detail"] or it["text"]
        for cat, pat, sev, plain in ANN_RULES:
            if re.search(pat, txt, re.I):
                quote = " ".join(txt.split()[:25])
                key = (cat, quote)
                if key in seen:
                    continue
                seen.add(key)
                flags.append({"category": cat, "severity": sev, "quote": quote,
                              "source_location": f"NSE announcement {it['date']}".strip(),
                              "plain_english": plain})
    return flags
