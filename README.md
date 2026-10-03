# Forensic Red-Flag Detector

Type an Indian company name (NSE/BSE) and get:
- Beneish M-Score and Altman Z-Score from annual statements (Yahoo Finance)
- Red flags from NSE announcements and an optional uploaded annual report (every quote verified verbatim, 25 words or fewer)
- SARIMA anomaly test on quarterly net income
- GARCH(1,1) volatility from daily prices
- Built-in Enron case study

## Files
- app.py: the website (Streamlit)
- data_sources.py: fetches data from Yahoo Finance and NSE; fails gracefully
- forensic_detector.py: Beneish, Altman, flag extraction, combined verdict
- forensic_timeseries.py: SARIMA and GARCH
- requirements.txt: libraries to install

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

Educational screening tool. Not investment or audit advice. Beneish and Altman are not meaningful for banks, NBFCs and insurers.
