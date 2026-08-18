# Stock Return Predictor (LSTM)

**(https://stock-return-predictor-4e2w4cjh9mibdywgk8iwse.streamlit.app/)

An LSTM-based model that predicts next-day stock returns from price-derived features (returns, moving averages). Includes rigorous backtesting against a naive baseline and statistical significance testing.

## Key Finding
The model achieves ~51.7% directional accuracy on held-out test data — not statistically distinguishable from a naive "predict no change" baseline (p=0.33). This is consistent with the efficient market hypothesis and reflects the genuine difficulty of short-term return prediction, rather than a modeling shortfall.

## What this project demonstrates
- LSTM architecture for time-series forecasting (PyTorch)
- Feature engineering (returns, moving averages)
- Proper train/test methodology and avoidance of the "naive persistence" evaluation trap
- Statistical rigor (binomial significance testing) rather than overselling results
- End-to-end deployment (Streamlit Community Cloud)

## Tech stack
Python, PyTorch, scikit-learn, pandas, yfinance, Streamlit
