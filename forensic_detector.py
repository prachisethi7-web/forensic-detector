#!/usr/bin/env python3
"""
Forensic Red-Flag Detector - Python backup
==========================================
Same logic as the web app:
  1. Rule-based extraction of qualitative red flags (every flag carries a verbatim quote, <=25 words)
  2. Beneish M-Score (8-variable) and Altman Z-Score (original manufacturing model)
  3. A combined screening verdict

Usage
-----
  python forensic_detector.py --sample                      # built-in Northwind demo
  python forensic_detector.py --text report.txt            # flags only
  python forensic_detector.py --text report.txt --fin fin.json --out result.json

fin.json format (same units for both periods):
  {"prior": {...15 fields...}, "current": {...15 fields...}}
Field names: receivables, sales, cogs, currentAssets, ppe, totalAssets, depreciation, sga,
             ltDebt, currentLiab, netIncome, ocf, totalLiab, retainedEarnings, marketEquity

No third-party packages needed (standard library only).
Educational screening tool, not investment or audit advice.
"""
import argparse
import json
import re

CATS = ["auditor_change", "related_party", "restatement", "off_balance_sheet",
        "going_concern", "management_turnover", "litigation"]
SEVERITIES = ["low", "medium", "high"]
MAX_QUOTE_WORDS = 25

# Words that tie an event to an accounting problem (used for turnover and litigation rules)
ACCT = re.compile(r"revenue recognition|restat|accounting|financial statements|audit committee|"
                  r"irregularit|internal control|misstat|false|misleading|improper", re.I)

# category -> detection pattern, severity rule, fixed plain-English sentence (no speculation)
RULESET = {
    "auditor_change": {
        "re": r"(dismissed|terminated|replaced|resigned)[^.]{0,80}(auditor|accounting firm|audit partners)"
              r"|(auditor|accounting firm)[^.]{0,60}(dismissed|terminated|resigned)"
              r"|engaged[^.]{0,60}(as|to serve as)[^.]{0,30}(auditor|accounting firm)",
        "high_if": r"disagree|irregular|restat|improper",
        "text": "The text describes a change in the company's independent auditor."},
    "related_party": {
        "re": r"related[- ]part(y|ies)|(owned|controlled) by[^.]{0,40}(chief|officer|director|CEO|CFO|COO|founder)"
              r"|general partner of",
        "high_if": r"not subject to competitive|no competitive|without (board|audit committee)"
                   r"|chief|officer|director|CFO|CEO|COO",
        "text": "The text describes a transaction or relationship with a related party."},
    "restatement": {
        "re": r"restat(e|ed|ement)|non-reliance|previously issued financial statements",
        "high_if": r"net income|revenue|reduc|overstat|error",
        "text": "The text describes a restatement of previously issued financial statements."},
    "off_balance_sheet": {
        "re": r"special[- ]purpose entit|variable interest entit|off[- ]balance|unconsolidated|not consolidated",
        "high_if": r"guarante",
        "text": "The text describes an entity or arrangement not consolidated on the balance sheet."},
    "going_concern": {
        "re": r"going concern|substantial doubt",
        "high_if": r".",
        "text": "The text includes going-concern or substantial-doubt language."},
    "management_turnover": {
        "re": r"(chief financial officer|CFO|controller|chief accounting officer|principal accounting officer)"
              r"[^.]{0,60}(resign|terminat|depart|stepped down|dismiss)",
        "needs_context": True,            # accounting term in previous/this/next sentence
        "high_if": r"investigat|restat|irregular",
        "text": "The text describes accounting-leadership turnover alongside an accounting matter."},
    "litigation": {
        "re": r"class action|lawsuit|complaint|subpoena|SEC (investigation|inquiry)|alleg",
        "needs_sentence": True,           # accounting term in the same sentence
        "high_if": r"class action|SEC",
        "text": "The text describes litigation or a regulatory action connected to accounting or reported results."},
}
HEADING = re.compile(r"^(note\s*\d+|item\s*\d|report of|liquidity|legal|management|risk|controls|going concern)", re.I)


def _norm(s):
    s = s.lower().replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", s).strip()


def extract_flags(text):
    """Return {company, period, flags[]} following the extractor schema."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    items, head = [], "not stated"
    for line in lines:
        if len(line) < 90 and not line.endswith(".") and HEADING.match(line):
            head = line
            continue
        for sent in re.split(r"(?<=[.!?])\s+", line):
            items.append((sent.strip(), head))

    flags, count = [], {}
    for i, (sent, heading) in enumerate(items):
        for cat, rule in RULESET.items():
            m = re.search(rule["re"], sent, re.I)
            if not m:
                continue
            context = " ".join(s for s, _ in items[max(i - 1, 0):i + 2])
            if rule.get("needs_context") and not ACCT.search(context):
                continue
            if rule.get("needs_sentence") and not ACCT.search(sent):
                continue
            count[cat] = count.get(cat, 0) + 1
            if count[cat] > 3:                       # cap per category to keep output readable
                continue
            quote = " ".join(sent[m.start():].split()[:MAX_QUOTE_WORDS])
            basis = context if rule.get("needs_context") else sent
            sev = "high" if re.search(rule["high_if"], basis, re.I) else "medium"
            flags.append({"category": cat, "severity": sev, "quote": quote,
                          "source_location": heading, "plain_english": rule["text"]})

    # Rule 2 / Rule 4 guard: keep only flags whose quote is verbatim and <=25 words
    source = _norm(text)
    flags = [f for f in flags if len(f["quote"].split()) <= MAX_QUOTE_WORDS and _norm(f["quote"]) in source]

    cm = re.match(r"^([A-Z][A-Za-z0-9 .,&'-]{2,60}?)\s+[\u2014\u2013-]", lines[0]) if lines else None
    pm = re.search(r"(fiscal year|FY)\s*\d{4}", text, re.I)
    return {"company": cm.group(1).strip() if cm else "not stated",
            "period": pm.group(0) if pm else "not stated",
            "flags": flags}


def beneish(a, b):
    """a = prior year (t-1), b = current year (t)."""
    dsri = (b["receivables"] / b["sales"]) / (a["receivables"] / a["sales"])
    gmi = ((a["sales"] - a["cogs"]) / a["sales"]) / ((b["sales"] - b["cogs"]) / b["sales"])
    aq = lambda x: 1 - (x["currentAssets"] + x["ppe"]) / x["totalAssets"]
    aqi = aq(b) / aq(a)
    sgi = b["sales"] / a["sales"]
    dep = lambda x: x["depreciation"] / (x["depreciation"] + x["ppe"])
    depi = dep(a) / dep(b)
    # If SG&A is not reported separately (common for Indian companies), use the neutral value 1.0
    sgai = (b["sga"] / b["sales"]) / (a["sga"] / a["sales"]) if a["sga"] and b["sga"] else 1.0
    tata = (b["netIncome"] - b["ocf"]) / b["totalAssets"]
    lv = lambda x: (x["ltDebt"] + x["currentLiab"]) / x["totalAssets"]
    lvgi = lv(b) / lv(a)
    m = (-4.84 + 0.920 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
         + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)
    return {"DSRI": dsri, "GMI": gmi, "AQI": aqi, "SGI": sgi, "DEPI": depi,
            "SGAI": sgai, "TATA": tata, "LVGI": lvgi, "M": m}


def altman(t):
    x1 = (t["currentAssets"] - t["currentLiab"]) / t["totalAssets"]
    x2 = t["retainedEarnings"] / t["totalAssets"]
    x3 = (t["sales"] - t["cogs"] - t["sga"]) / t["totalAssets"]   # EBIT proxy = sales - COGS - SG&A
    x4 = t["marketEquity"] / t["totalLiab"]
    x5 = t["sales"] / t["totalAssets"]
    return {"X1": x1, "X2": x2, "X3": x3, "X4": x4, "X5": x5,
            "Z": 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + x5}


def m_label(m):
    return "Likely manipulator" if m > -1.78 else "Watch list" if m > -2.22 else "Low probability"


def z_label(z):
    return "Safe zone" if z > 2.99 else "Grey zone" if z >= 1.81 else "Distress zone"


def combined_verdict(flags, m=None, z=None):
    high = sum(f["severity"] == "high" for f in flags)
    quant = (m is not None and m > -1.78) or (z is not None and z < 1.81)
    if (high >= 1 and quant) or high >= 2:
        return "HIGH CONCERN"
    if high >= 1 or len(flags) >= 2 or quant or (m is not None and m > -2.22):
        return "ELEVATED"
    return "NO RED FLAGS FOUND"


SAMPLE_TEXT = """NORTHWIND INDUSTRIAL CORP. \u2014 Annual Report, Fiscal Year 2025 (excerpts)

Report of Independent Registered Public Accounting Firm
On March 4, 2025, the Audit Committee dismissed Halvorsen & Pike LLP as the Company's independent auditor following a disagreement regarding the timing of revenue recognition on bill-and-hold arrangements. Brightwater Audit Partners was engaged on March 20, 2025.

Note 7 \u2014 Related Party Transactions
During fiscal 2025, the Company purchased raw materials totaling $41.2 million from Kestrel Supply LLC, an entity wholly owned by the Company's Chief Operating Officer. These purchases were not subject to competitive bidding.

Note 12 \u2014 Restatement of Previously Issued Financial Statements
The Company has restated its consolidated financial statements for the fiscal year ended December 31, 2024, to correct errors in the recognition of revenue. The restatement reduced previously reported net income by $18.6 million.

Note 14 \u2014 Variable Interest Entities
The Company holds a variable interest in Harbor Point Funding Trust, which is not consolidated in these financial statements. The Company has guaranteed $65 million of the Trust's borrowings.

Liquidity
These conditions raise substantial doubt about the Company's ability to continue as a going concern within one year after the date the financial statements are issued.

Management Changes
In June 2025, the Chief Financial Officer resigned effective immediately. The Audit Committee's investigation into the revenue recognition matters described in Note 12 is ongoing.

Legal Proceedings
In September 2025, a putative shareholder class action was filed alleging that the Company issued materially false statements about its reported revenue.

Separately, our Vice President of Marketing retired in August 2025 after twenty-two years with the Company."""

SAMPLE_FIN = {
    "prior":   dict(receivables=100, sales=1000, cogs=600, currentAssets=400, ppe=300, totalAssets=900,
                    depreciation=40, sga=150, ltDebt=150, currentLiab=200, netIncome=80, ocf=90,
                    totalLiab=400, retainedEarnings=250, marketEquity=800),
    "current": dict(receivables=220, sales=1300, cogs=880, currentAssets=520, ppe=330, totalAssets=1200,
                    depreciation=35, sga=200, ltDebt=300, currentLiab=280, netIncome=150, ocf=40,
                    totalLiab=700, retainedEarnings=300, marketEquity=600),
}


def analyze(text=None, fin=None):
    result = extract_flags(text) if text else {"company": "not stated", "period": "not stated", "flags": []}
    m = z = None
    scores = {}
    if fin:
        bm, am = beneish(fin["prior"], fin["current"]), altman(fin["current"])
        m, z = bm["M"], am["Z"]
        scores = {"beneish": {k: round(v, 4) for k, v in bm.items()}, "beneish_label": m_label(m),
                  "altman": {k: round(v, 4) for k, v in am.items()}, "altman_label": z_label(z)}
    return {**result, "scores": scores, "combined_verdict": combined_verdict(result["flags"], m, z)}


def main():
    p = argparse.ArgumentParser(description="Forensic Red-Flag Detector")
    p.add_argument("--text", help="path to a .txt file with disclosure text")
    p.add_argument("--fin", help="path to a JSON file with prior/current financials")
    p.add_argument("--sample", action="store_true", help="run the built-in Northwind demo")
    p.add_argument("--out", help="write the full result as JSON to this path")
    p.add_argument("--timeseries", action="store_true",
                   help="also run the Enron SARIMA anomaly test (needs statsmodels; see forensic_timeseries.py)")
    args = p.parse_args()

    if args.timeseries and not (args.sample or args.text or args.fin):
        import forensic_timeseries
        forensic_timeseries.main()
        return

    if args.sample:
        text, fin = SAMPLE_TEXT, SAMPLE_FIN
    else:
        text = open(args.text, encoding="utf-8").read() if args.text else None
        fin = json.load(open(args.fin)) if args.fin else None
    if not text and not fin:
        p.error("provide --text and/or --fin, or use --sample")

    res = analyze(text, fin)
    print(f"Company: {res['company']}   Period: {res['period']}")
    print(f"Combined verdict: {res['combined_verdict']}")
    if res["scores"]:
        print(f"Beneish M = {res['scores']['beneish']['M']:.2f} ({res['scores']['beneish_label']})")
        print(f"Altman  Z = {res['scores']['altman']['Z']:.2f} ({res['scores']['altman_label']})")
    print(f"\n{len(res['flags'])} flag(s):")
    for f in res["flags"]:
        print(f"  [{f['severity'].upper():6}] {f['category']:<20} {f['source_location']}")
        print(f"           \"{f['quote']}\"")
    if args.timeseries:
        import forensic_timeseries
        oos = forensic_timeseries.sarima_out_of_sample()
        print("\nSARIMA out-of-sample check (Enron example): flagged", oos.index[oos.anomalous].tolist() or "none")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
