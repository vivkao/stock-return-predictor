import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle

device = torch.device('cpu')  # Streamlit Cloud has no GPU

# --- Model definition (must match your training code exactly) ---
class PredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(PredictionModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def inverse_transform_return(scaled_return_col, scaler, n_features, return_col_idx=0):
    dummy = np.zeros((len(scaled_return_col), n_features))
    dummy[:, return_col_idx] = scaled_return_col.flatten()
    return scaler.inverse_transform(dummy)[:, return_col_idx]

def reconstruct_prices(starting_price, returns):
    prices = [starting_price]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    return prices

# --- Load saved model + scaler once ---
@st.cache_resource
def load_model_and_scaler():
    feature_cols = ['Return', 'MA_7', 'MA_30']
    model = PredictionModel(input_dim=len(feature_cols), hidden_dim=32, num_layers=2, output_dim=1)
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.eval()
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler, feature_cols

model, scaler, feature_cols = load_model_and_scaler()

# --- Streamlit UI ---
st.title("Stock Return Predictor (LSTM)")
st.write("Predicts next-day stock returns using an LSTM trained on price-derived features. "
         "Note: directional accuracy is close to a naive baseline (~51-52%), reflecting the "
         "genuine difficulty of short-term return prediction.")

ticker = st.text_input("Enter a stock ticker", value="AAPL")
st.caption("Note: this demo re-evaluates on a recent window for illustration. "
           "In rigorous backtesting (proper train/test split + statistical significance testing), "
           "this model achieved ~51.7% directional accuracy — not statistically distinguishable "
           "from a naive baseline. See the GitHub repo for full methodology.")

if st.button("Run Prediction"):
    with st.spinner("Downloading data and running model..."):
        df = yf.download(ticker, start="2015-01-01")

        if df.empty:
            st.error("No data found for that ticker.")
        else:
            df['Return'] = df['Close'].pct_change()
            df['MA_7'] = df['Close'].rolling(window=7).mean()
            df['MA_30'] = df['Close'].rolling(window=30).mean()
            df = df.dropna()

            scaled_data = scaler.transform(df[feature_cols])

            seq_length = 30
            data = []
            for i in range(len(scaled_data) - seq_length):
                data.append(scaled_data[i:i+seq_length])
            data = np.array(data)

            window_size = 300
            X = torch.from_numpy(data[-window_size:, :-1, :]).float()
            y_scaled = data[-window_size:, -1, 0]

            with torch.no_grad():
                y_pred_scaled = model(X).numpy()

            n_features = len(feature_cols)
            y_pred_actual = inverse_transform_return(y_pred_scaled, scaler, n_features)
            y_actual = inverse_transform_return(y_scaled.reshape(-1, 1), scaler, n_features)

            start_idx = len(df) - len(y_actual)
            starting_price = df['Close'].iloc[start_idx - 1]

            actual_prices = reconstruct_prices(starting_price, y_actual)[1:]
            predicted_prices = reconstruct_prices(starting_price, y_pred_actual)[1:]

            actual_direction = np.sign(y_actual)
            predicted_direction = np.sign(y_pred_actual)
            directional_accuracy = (actual_direction == predicted_direction).mean()

            st.metric("Directional Accuracy", f"{directional_accuracy:.2%}")

            fig, ax = plt.subplots(figsize=(12, 5))
            dates = df.iloc[-len(actual_prices):].index
            ax.plot(dates, actual_prices, color='blue', label='Actual Price')
            ax.plot(dates, predicted_prices, color='green', label='Predicted Price')
            ax.legend()
            ax.set_title(f"{ticker} Price (reconstructed from predicted returns)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            st.pyplot(fig)
            st.caption("The diverging green line illustrates compounding drift: small daily prediction "
           "biases compound multiplicatively over time. See return-level chart below for a "
           "fairer day-by-day comparison.")

            fig2, ax2 = plt.subplots(figsize=(12, 4))
            ax2.plot(dates, y_actual, color='blue', label='Actual Return', alpha=0.7)
            ax2.plot(dates, y_pred_actual, color='green', label='Predicted Return', alpha=0.7)
            ax2.axhline(0, color='gray', linestyle=':')
            ax2.legend()
            ax2.set_title(f"{ticker} Daily Return Prediction")
            st.pyplot(fig2)