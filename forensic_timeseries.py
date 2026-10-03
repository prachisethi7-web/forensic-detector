#!/usr/bin/env python3
"""
Forensic Red-Flag Detector - time-series evidence module
========================================================
Combines the two scripts supplied with the project:
  * SARIMA anomaly detection on Enron quarterly net income   (sarima_enron_anomaly.py)
  * GARCH(1,1) volatility analysis on Infosys share prices   (garch_infosys_volatility.py)

Changes from the original scripts
  * SARIMA: quarter labels fixed (original printed '2001-Q%q').
  * SARIMA: adds an out-of-sample test (fit on 1999Q1-2001Q2, forecast 2001Q3-Q4), which is
    the cleaner method suggested in the original script's own limitations note.
  * GARCH: returns are forced to a 1-column Series (newer yfinance returns a DataFrame).
  * GARCH: failed downloads now say so, instead of claiming the company is delisted.
  * yfinance / arch are imported only when GARCH is used, so SARIMA runs without them.

Usage
  python forensic_timeseries.py sarima                         # built-in Enron example
  python forensic_timeseries.py sarima --name "Acme" --start 2016Q1 \\
        --values "120,131,128,140,...,95,60"                    # ANY company: your own quarterly net income
  python forensic_timeseries.py garch --ticker TCS.NS --name "TCS"   # ANY listed company (needs internet)
  python forensic_timeseries.py garch --delisted --name "Enron"      # fallback for delisted companies

Requires: pandas numpy matplotlib statsmodels   (GARCH also: yfinance arch)
Educational tool, not investment or audit advice.
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Enron quarterly net income ($M), 1999Q1-2001Q4. Source: user-provided figures (see project notes:
# Enron did not file a normal Q4-2001 report, so verify the 2001Q4 value before relying on it).
QUARTERLY_NET_INCOME = [253, 201, 201, 231, 338, 289, 160, 193, 425, 404, -638, -809]
ORDER, SEASONAL_ORDER = (1, 1, 1), (0, 1, 0, 4)
Z_THRESHOLD = 2.0


def quarter_label(ts):
    return f"{ts.year}-Q{(ts.month - 1) // 3 + 1}"


def enron_series():
    idx = pd.date_range("1999-01-01", periods=len(QUARTERLY_NET_INCOME), freq="QS")
    return pd.Series(QUARTERLY_NET_INCOME, index=idx, name="net_income")


def _fit(series):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    return SARIMAX(series, order=ORDER, seasonal_order=SEASONAL_ORDER,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)


def sarima_in_sample(series=None):
    """Original method: one-step-ahead predictions over the whole sample."""
    series = enron_series() if series is None else series
    fit = _fit(series)
    pred = fit.get_prediction(start=1, end=len(series) - 1)
    mean, ci = pred.predicted_mean, pred.conf_int(alpha=0.05)
    resid = series.iloc[1:] - mean
    z = (resid - resid.mean()) / resid.std()
    df = pd.DataFrame({"actual": series.iloc[1:], "predicted": mean, "lower_95": ci.iloc[:, 0],
                       "upper_95": ci.iloc[:, 1], "residual": resid, "z_score": z})
    df["outside_interval"] = (df.actual < df.lower_95) | (df.actual > df.upper_95)
    df["z_flag"] = df.z_score.abs() > Z_THRESHOLD
    df["anomalous"] = df.outside_interval | df.z_flag
    df.index = [quarter_label(t) for t in df.index]
    return df


def sarima_out_of_sample(series=None, train_quarters=10):
    """Cleaner method: fit before the break (1999Q1-2001Q2), forecast the rest."""
    series = enron_series() if series is None else series
    train, test = series.iloc[:train_quarters], series.iloc[train_quarters:]
    fc = _fit(train).get_forecast(steps=len(test))
    ci = fc.conf_int(alpha=0.05)
    df = pd.DataFrame({"actual": test, "predicted": fc.predicted_mean,
                       "lower_95": ci.iloc[:, 0].values, "upper_95": ci.iloc[:, 1].values})
    df["error"] = df.actual - df.predicted
    df["anomalous"] = (df.actual < df.lower_95) | (df.actual > df.upper_95)
    df.index = [quarter_label(t) for t in df.index]
    return df


def plot_sarima(df, path="enron_sarima_anomalies.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = enron_series()
    labels = [quarter_label(t) for t in s.index]
    plt.figure(figsize=(11, 5))
    plt.plot(labels, s.values, "ko-", label="Actual net income")
    plt.plot(df.index, df.predicted, "--", color="steelblue", label="SARIMA one-step prediction")
    plt.fill_between(df.index, df.lower_95, df.upper_95, color="steelblue", alpha=0.15, label="95% interval")
    f = df[df.anomalous]
    plt.scatter(f.index, f.actual, color="crimson", s=100, zorder=5, label="Flagged")
    plt.axhline(0, color="grey", lw=0.8)
    plt.xticks(rotation=45)
    plt.title("Enron quarterly net income - SARIMA anomaly detection")
    plt.ylabel("Net income ($M)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ----------------------------------------------------------------------------- GARCH
def fetch_returns(ticker, period="5y"):
    import yfinance as yf
    try:
        data = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
        if data is None or data.empty or "Close" not in data:
            return None
        close = data["Close"].squeeze().dropna()          # 1-column DataFrame -> Series
        if len(close) < 60:
            return None
        return (100 * np.log(close / close.shift(1)).dropna()).rename("returns")
    except Exception as exc:
        print(f"[fetch_returns] could not fetch {ticker}: {exc}")
        return None


def garch_fit(ticker="INFY.NS", name="Infosys Ltd"):
    """Returns (summary dict, annualized conditional volatility Series or None)."""
    from arch import arch_model
    returns = fetch_returns(ticker)
    if returns is None:
        return ({"company": name, "status": "FETCH_FAILED",
                 "note": "Price download failed (network, rate limit or wrong ticker). Try again later."}, None)
    res = arch_model(returns, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    ann = res.conditional_volatility * np.sqrt(252)
    fc = np.sqrt(res.forecast(horizon=10, reindex=False).variance.values[-1, :])
    return ({"company": name, "status": "FITTED",
             "latest_annualized_vol_pct": round(float(ann.iloc[-1]), 2),
             "mean_annualized_vol_pct": round(float(ann.mean()), 2),
             "max_annualized_vol_pct": round(float(ann.max()), 2),
             "10d_forecast_daily_vol_pct": np.round(fc, 3).tolist()}, ann)


def garch_summary(ticker="INFY.NS", name="Infosys Ltd"):
    return garch_fit(ticker, name)[0]


def garch_enron_fallback():
    return {"company": "Enron Corp", "status": "NO_LIVE_PRICE_SERIES",
            "note": "Enron was delisted after its Dec-2001 Chapter 11 filing; no usable live series. "
                    "The 2001 collapse was a discrete default event, not a volatility regime GARCH can model."}


# ----------------------------------------------------------------------------- CLI
def custom_series(values, start):
    """Build a quarterly series from a comma-separated string like '253,201,...' and a start like 1999Q1."""
    vals = [float(v) for v in values.replace(" ", "").split(",") if v]
    y, q = int(start[:4]), int(start[-1])
    idx = pd.date_range(f"{y}-{3 * (q - 1) + 1:02d}-01", periods=len(vals), freq="QS")
    return pd.Series(vals, index=idx, name="net_income")


def run_sarima(series, name, train_quarters=None, as_json=False):
    n = len(series)
    if n < 8:
        raise SystemExit(f"Need at least 8 quarters for SARIMA(1,1,1)x(0,1,0,4); got {n}.")
    if n < 20:
        print(f"WARNING: only {n} quarters. Results are illustrative; 40+ is recommended.\n")
    train_quarters = train_quarters or n - 2
    ins = sarima_in_sample(series)
    oos = sarima_out_of_sample(series, train_quarters)
    if as_json:
        print(json.dumps({"company": name, "in_sample": ins.round(2).reset_index().to_dict("records"),
                          "out_of_sample": oos.round(2).reset_index().to_dict("records")}, indent=2))
        return
    pd.set_option("display.width", 160)
    print(f"SARIMA anomaly test - {name}\n")
    print("IN-SAMPLE one-step-ahead\n", ins.round(1).to_string())
    print("\nFlagged:", ins.index[ins.anomalous].tolist() or "none")
    print(f"\nOUT-OF-SAMPLE (fit first {train_quarters} quarters, forecast the remaining {n - train_quarters})\n",
          oos.round(1).to_string())
    print("\nFlagged:", oos.index[oos.anomalous].tolist() or "none")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Time-series evidence for any company")
    p.add_argument("cmd", nargs="?", default="sarima", choices=["sarima", "garch"])
    p.add_argument("--name", default=None, help="company name for the report")
    p.add_argument("--values", help="sarima: comma-separated quarterly net income, oldest first")
    p.add_argument("--start", default="1999Q1", help="sarima: first quarter of --values, e.g. 2018Q1")
    p.add_argument("--train", type=int, help="sarima: quarters used for the out-of-sample fit (default n-2)")
    p.add_argument("--ticker", default="INFY.NS", help="garch: Yahoo Finance ticker, e.g. TCS.NS, AAPL")
    p.add_argument("--delisted", action="store_true", help="garch: skip the fit (company no longer trades)")
    p.add_argument("--json", action="store_true")
    args, _ = p.parse_known_args()
    if args.cmd == "sarima":
        if args.values:
            series, name = custom_series(args.values, args.start), args.name or "Custom company"
        else:
            series, name = enron_series(), args.name or "Enron Corp (built-in example)"
            plot_sarima(sarima_in_sample(series))
        run_sarima(series, name, args.train, args.json)
    else:
        name = args.name or args.ticker
        res = garch_enron_fallback() if args.delisted else garch_summary(args.ticker, name)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
