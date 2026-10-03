"""
Forensic Red-Flag Detector - Streamlit app
Search one company (NSE/BSE) and get: Beneish + Altman scores, red flags, SARIMA anomaly test, GARCH volatility.
Run locally:  streamlit run app.py
"""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import data_sources as ds
import forensic_detector as fd
import forensic_timeseries as ft

st.set_page_config(page_title="Forensic Red-Flag Detector", page_icon="🔎", layout="wide")


# ------------------------------------------------------------------ cached fetchers
@st.cache_data(ttl=3600, show_spinner=False)
def c_search(q): return ds.search_company(q)
@st.cache_data(ttl=3600, show_spinner=False)
def c_fin(sym): return ds.get_financials(sym)
@st.cache_data(ttl=3600, show_spinner=False)
def c_qni(sym):
    s = ds.get_quarterly_net_income(sym)
    return None if s is None else s
@st.cache_data(ttl=1800, show_spinner=False)
def c_ann(sym): return ds.get_announcements(sym)
@st.cache_data(ttl=3600, show_spinner=False)
def c_garch(sym, name): return ft.garch_fit(sym, name)


# ------------------------------------------------------------------ scoring with graceful gaps
BEN_REQ = ["receivables", "sales", "cogs", "currentAssets", "ppe", "totalAssets", "depreciation",
           "ltDebt", "currentLiab", "netIncome", "ocf"]
ALT_REQ = ["currentAssets", "currentLiab", "totalAssets", "retainedEarnings", "sales", "cogs",
           "marketEquity", "totalLiab"]


def compute_scores(fin):
    p, c = dict(fin["prior"]), dict(fin["current"])
    out = {"bm": None, "am": None, "notes": [], "missing_b": [], "missing_a": []}
    if p.get("sga") is None or c.get("sga") is None:
        p["sga"] = c["sga"] = 0.0
        out["notes"].append("SG&A is not reported separately; SGAI is set to a neutral 1.0 and Altman EBIT = sales - COGS (overstated).")
    out["missing_b"] = [k for k in BEN_REQ if p.get(k) is None or c.get(k) is None]
    out["missing_a"] = [k for k in ALT_REQ if c.get(k) is None]
    if not out["missing_b"]:
        try:
            out["bm"] = fd.beneish(p, c)
        except (ZeroDivisionError, TypeError):
            out["notes"].append("Beneish could not be computed (a figure that must be divided by was zero).")
    if not out["missing_a"]:
        try:
            out["am"] = fd.altman(c)
        except (ZeroDivisionError, TypeError):
            out["notes"].append("Altman could not be computed (zero denominator).")
    return out


def pdf_to_text(file):
    from pypdf import PdfReader
    pages = PdfReader(file).pages[:250]
    raw = "\n".join((pg.extract_text() or "") for pg in pages)
    return re.sub(r"(?<![.:\n])\n(?!\n)", " ", raw)


def flag_df(flags):
    return pd.DataFrame(flags)[["category", "severity", "quote", "source_location", "plain_english"]]


def stamp(label):
    color = {"HIGH CONCERN": "#9b2226", "ELEVATED": "#a9721f"}.get(label, "#3a6b35")
    st.markdown(f"<div style='display:inline-block;border:4px solid {color};color:{color};border-radius:8px;"
                f"padding:10px 18px;font:700 24px monospace;transform:rotate(-2deg)'>{label}</div>",
                unsafe_allow_html=True)


# ------------------------------------------------------------------ SARIMA + GARCH panels
def sarima_panel(key, auto_series=None, preset=None):
    series = preset
    if preset is not None:
        pass
    elif auto_series is not None and len(auto_series) >= 8:
        series = pd.Series(auto_series.values.astype(float),
                           index=pd.date_range(str(auto_series.index[0])[:10], periods=len(auto_series), freq="QS"))
        st.success(f"Using {len(series)} quarters of net income from Yahoo Finance.")
    else:
        if auto_series is not None:
            st.info(f"Yahoo returned only {len(auto_series)} quarter(s); SARIMA needs at least 8 (ideally 40+). "
                    "Paste a longer quarterly net income history below (e.g. from the company's results or Screener.in).")
        c1, c2 = st.columns([3, 1])
        vals = c1.text_area("Quarterly net income, oldest first, comma-separated (same units)", key=key + "v",
                            placeholder="120,131,128,140,...")
        start = c2.text_input("First quarter", "2019Q1", key=key + "s", help="Format YYYYQn, e.g. 2019Q1 (calendar quarters)")
        if vals.strip():
            try:
                series = ft.custom_series(vals, start)
            except Exception:
                st.error("Could not read those numbers. Use plain numbers separated by commas, and a start like 2019Q1.")
    if series is None:
        return
    if len(series) < 8:
        st.warning("Need at least 8 quarters.")
        return
    if len(series) < 20:
        st.warning(f"Only {len(series)} quarters: treat results as illustrative, not reliable.")
    try:
        ins = ft.sarima_in_sample(series)
        oos = ft.sarima_out_of_sample(series, len(series) - 2)
    except Exception as e:
        st.error(f"SARIMA could not be fitted on this data ({e}).")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = [ft.quarter_label(t) for t in series.index]
    ax.plot(labels, series.values, "ko-", label="Actual")
    ax.plot(ins.index, ins.predicted, "--", color="steelblue", label="SARIMA one-step prediction")
    f = ins[ins.anomalous]
    ax.scatter(f.index, f.actual, color="crimson", s=80, zorder=5, label="Flagged (in-sample)")
    f2 = oos[oos.anomalous]
    ax.scatter(f2.index, f2.actual, color="darkred", marker="X", s=110, zorder=6, label="Flagged (out-of-sample)")
    ax.axhline(0, color="grey", lw=0.8)
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    ax.legend(fontsize=8)
    st.pyplot(fig)
    st.markdown("**Out-of-sample test** (fit on all but the last 2 quarters, forecast those 2) - the cleaner method")
    st.dataframe(oos.round(1), width="stretch")
    st.markdown("**In-sample one-step-ahead test** (can wrongly flag the first few quarters)")
    st.dataframe(ins.round(1), width="stretch")
    st.caption("SARIMA(1,1,1)x(0,1,0,4). Flags a quarter if it falls outside the 95% interval (or |z| > 2 in-sample).")


def garch_panel(symbol, name, delisted=False):
    if delisted:
        st.info(ft.garch_enron_fallback()["note"])
        return None
    with st.spinner("Downloading prices and fitting GARCH(1,1)..."):
        try:
            summary, ann = c_garch(symbol, name)
        except Exception as e:
            st.error(f"GARCH failed: {e}")
            return None
    if summary["status"] != "FITTED":
        st.warning(summary["note"])
        return None
    a, b, c = st.columns(3)
    a.metric("Latest annualized volatility", f"{summary['latest_annualized_vol_pct']}%")
    b.metric("5-year average", f"{summary['mean_annualized_vol_pct']}%")
    c.metric("5-year peak", f"{summary['max_annualized_vol_pct']}%")
    st.line_chart(ann.rename("Annualized volatility (%)"))
    st.markdown("10-day forecast of daily volatility (%)")
    st.bar_chart(pd.Series(summary["10d_forecast_daily_vol_pct"], index=range(1, 11)))
    ratio = summary["latest_annualized_vol_pct"] / summary["mean_annualized_vol_pct"]
    st.caption(f"Latest / average = {ratio:.2f}. GARCH measures how jumpy daily returns are; it does not by itself indicate fraud.")
    return summary


# ------------------------------------------------------------------ pages
def page_company():
    st.subheader("Search a company")
    ex = st.columns(5)
    for col, n in zip(ex, ["Infosys", "TCS", "Reliance Industries", "Wipro", "Satyam"]):
        if col.button(n, width="stretch"):
            st.session_state["q"] = n
    q = st.text_input("Company name or NSE/BSE ticker", key="q", placeholder="e.g. Infosys, TCS, Tata Motors")
    if not q.strip():
        st.stop()
    with st.spinner("Searching..."):
        hits = c_search(q)
    if not hits:
        st.error("No NSE/BSE match found. Try the exact ticker (e.g. TCS) or a different spelling.")
        st.stop()
    labels = [f"{h['name']}  ({h['symbol']}, {h['exchange']})" for h in hits]
    pick = hits[labels.index(st.selectbox("Match", labels))] if len(hits) > 1 else hits[0]
    sym, name = pick["symbol"], pick["name"]
    st.markdown(f"### {name} `{sym}`")

    with st.spinner("Fetching financial statements..."):
        fin = c_fin(sym)
    if fin:
        import copy
        fin = copy.deepcopy(fin)
        if fin["current"].get("marketEquity") is None:
            st.warning("Yahoo did not return a market capitalisation, which Altman's X4 needs.")
            mc_cr = st.number_input("Optional: enter market cap in Rs crore (find it on NSE, Screener or Google Finance)",
                                    min_value=0.0, value=0.0, step=100.0, format="%.1f")
            if mc_cr > 0:
                fin["current"]["marketEquity"] = mc_cr * 1e7          # 1 crore = 10,000,000
                fin["market_cap_source"] = "entered manually"
        elif fin.get("market_cap_source"):
            st.caption(f"Market cap source: {fin['market_cap_source']}")
    scores = compute_scores(fin) if fin else None
    with st.spinner("Checking NSE announcements..."):
        ann = c_ann(sym)
    ann_flags = ds.classify_announcements(ann)

    st.markdown("**Optional: add the annual report for deeper text flags**")
    up = st.file_uploader("Upload annual report (PDF or TXT) or paste text below", type=["pdf", "txt"])
    pasted = st.text_area("...or paste disclosure text", height=100)
    text = ""
    if up is not None:
        try:
            text = pdf_to_text(up) if up.name.lower().endswith(".pdf") else up.read().decode("utf-8", "ignore")
        except Exception as e:
            st.error(f"Could not read that file ({e}).")
    text = (text + "\n" + pasted).strip()
    text_flags = fd.extract_flags(text)["flags"] if text else []
    flags = ann_flags + text_flags

    M = scores["bm"]["M"] if scores and scores["bm"] else None
    Z = scores["am"]["Z"] if scores and scores["am"] else None

    t1, t2, t3, t4, t5 = st.tabs(["Verdict", "Scores", "Red flags", "SARIMA", "GARCH"])
    with t1:
        stamp(fd.combined_verdict(flags, M, Z))
        c = st.columns(4)
        c[0].metric("Beneish M", "n/a" if M is None else f"{M:.2f}", None if M is None else fd.m_label(M), delta_color="off")
        c[1].metric("Altman Z", "n/a" if Z is None else f"{Z:.2f}", None if Z is None else fd.z_label(Z), delta_color="off")
        c[2].metric("Red flags", len(flags))
        c[3].metric("High severity", sum(f["severity"] == "high" for f in flags))
        st.caption("Screening heuristic only: not an audit conclusion or investment advice.")
        if M is None and Z is None:
            st.warning("Scores unavailable: financial statements were missing or incomplete from Yahoo Finance.")
    with t2:
        if not fin:
            st.error("Could not fetch financial statements for this ticker.")
        else:
            st.caption(f"Annual periods compared: {fin['periods'][0]} (prior) vs {fin['periods'][1]} (current). "
                       "Market equity uses today's market capitalisation (approximation).")
            for n in scores["notes"]:
                st.info(n)
            if scores["missing_b"]:
                st.warning("Beneish skipped, missing: " + ", ".join(scores["missing_b"]))
            if scores["missing_a"]:
                st.warning("Altman skipped, missing: " + ", ".join(scores["missing_a"]))
            if scores["bm"]:
                st.markdown("**Beneish M-Score components**")
                st.dataframe(pd.Series(scores["bm"]).round(3).rename("value").to_frame(), width="stretch")
            if scores["am"]:
                st.markdown("**Altman Z-Score components**")
                st.dataframe(pd.Series(scores["am"]).round(3).rename("value").to_frame(), width="stretch")
            st.caption("Beneish/Altman were built for manufacturing-style companies; results for banks, NBFCs and insurers are not meaningful.")
    with t3:
        if ann is None:
            st.warning("NSE announcements could not be fetched (the exchange often blocks cloud servers). "
                       "Upload or paste the annual report above for text-based flags.")
        else:
            st.caption(f"Checked {len(ann)} recent NSE announcements.")
        if flags:
            st.dataframe(flag_df(flags), width="stretch")
        else:
            st.success("No quote-verified flags found in the data supplied.")
        st.caption("Announcement flags quote the exchange subject line. Management-turnover flags from announcements are 'low' "
                   "because the notice does not state an accounting reason.")
    with t4:
        sarima_panel("sar", c_qni(sym))
    with t5:
        garch_panel(sym, name)


def page_enron():
    st.subheader("Built-in case study: Enron Corp. (FY2000)")
    stamp("HIGH CONCERN")
    c = st.columns(3)
    c[0].metric("Beneish M (FY2000 vs FY1999)", "-0.80", "Likely manipulator", delta_color="off")
    c[1].metric("Altman Z", "2.23", "Grey zone", delta_color="off")
    c[2].metric("SARIMA flags", "2001-Q3, Q4", "out-of-sample", delta_color="off")
    st.caption("Scores copied from the project's Case Files (SEC filings). Qualitative items below are a public-record summary, not extractor output.")
    st.markdown("- Special-purpose entities (LJM1, LJM2, Raptors) used to move debt and losses off the balance sheet.\n"
                "- CFO Andrew Fastow was general partner of LJM partnerships that transacted with Enron.\n"
                "- November 2001 restatement of 1997-2000 results after SPEs were reconsolidated.")
    st.markdown("#### SARIMA - quarterly net income")
    st.caption("Figures user-provided; Enron did not file a normal Q4-2001 report, so treat 2001-Q4 with caution.")
    sarima_panel("enron", preset=ft.enron_series())
    st.markdown("#### GARCH")
    garch_panel("ENE", "Enron Corp", delisted=True)


def page_how():
    st.subheader("How it works")
    st.markdown(
        "**Data:** Yahoo Finance (annual statements, prices) and NSE announcements; optional annual report upload.\n\n"
        "**Beneish M-Score:** 8 ratios comparing two years; above -1.78 suggests possible manipulation.\n\n"
        "**Altman Z-Score:** 5 ratios; below 1.81 is the distress zone.\n\n"
        "**Red flags:** pattern rules; every quote is checked to be verbatim and 25 words or fewer.\n\n"
        "**SARIMA:** forecasts quarterly net income and flags quarters far outside the forecast.\n\n"
        "**GARCH(1,1):** measures how volatile daily returns are over time.\n\n"
        "**Limits:** free data can be incomplete; the models were designed for ordinary companies, not banks; "
        "results are a screening aid, not proof of wrongdoing.")


# ------------------------------------------------------------------ layout
st.title("🔎 Forensic Red-Flag Detector")
st.caption("Type a company. Get accounting-quality scores, red flags, a SARIMA anomaly test and GARCH volatility.")
mode = st.sidebar.radio("Mode", ["Search a company", "Enron case study", "How it works"])
{"Search a company": page_company, "Enron case study": page_enron, "How it works": page_how}[mode]()
