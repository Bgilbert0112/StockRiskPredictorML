import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox, ttk
from matplotlib import dates as mdates
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler

def get_stock_data(tickers, start_date="2023-01-01", end_date="2025-02-25"):
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if data.empty:
            raise ValueError("No data returned for the given tickers.")
        if "Adj Close" not in data.columns:
            if "Close" in data.columns:
                return data["Close"]
            else:
                raise ValueError("Neither 'Adj Close' nor 'Close' data available.")
        return data["Adj Close"]
    except Exception as e:
        raise Exception(f"Error fetching data: {str(e)}")

def calculate_returns(prices):
    returns = prices.pct_change().dropna()
    return returns

def prepare_lstm_data(returns, lookback=30):
    scaler = MinMaxScaler()
    scaled_returns = scaler.fit_transform(returns.values.reshape(-1, 1))
    X, y = [], []
    for i in range(lookback, len(scaled_returns)):
        X.append(scaled_returns[i-lookback:i])
        y.append(scaled_returns[i])
    return np.array(X), np.array(y), scaler

def train_lstm(X, y):
    model = Sequential()
    model.add(Input(shape=(X.shape[1], 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    return model

def predict_volatility(returns, weights, lookback=30):
    portfolio_returns = (returns * weights).sum(axis=1)
    X, _, scaler = prepare_lstm_data(portfolio_returns, lookback)
    model = train_lstm(X[:-1], X[1:, 0])
    last_sequence = X[-1].reshape(1, lookback, 1)
    pred = model.predict(last_sequence, verbose=0)
    pred_vol = scaler.inverse_transform(pred)[0][0]
    return abs(pred_vol)

def calculate_var(returns, weights, confidence_level=0.95, horizon=30):
    portfolio_returns = (returns * weights).sum(axis=1)
    var = np.percentile(portfolio_returns, (1 - confidence_level) * 100) * np.sqrt(horizon)
    return abs(var)  # Return absolute value for positive VaR

def predict_percentage_move(returns, weights, lookback=30, days_ahead=30):
    portfolio_returns = (returns * weights).sum(axis=1)
    X, _, scaler = prepare_lstm_data(portfolio_returns, lookback)

    model = Sequential()
    model.add(Input(shape=(lookback, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, portfolio_returns.values[lookback:], epochs=10, batch_size=32, verbose=0)

    last_sequence = X[-1].reshape(1, lookback, 1)
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(days_ahead):
        next_pred = model.predict(current_sequence, verbose=0)
        future_predictions.append(next_pred[0, 0])
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = next_pred[0, 0]

    avg_return = np.mean(future_predictions)
    percentage_move = avg_return * 100  # Convert to percentage
    return percentage_move

class PortfolioRiskApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Portfolio Risk Predictor")
        self.root.geometry("1000x1200")  # Larger window size

        # Dark mode styling
        self.root.configure(bg="#2c2c2c")  # Dark gray background
        style = ttk.Style()
        style.theme_use("clam")  # Use a theme that supports customization
        style.configure("TLabel", background="#2c2c2c", foreground="white")
        style.configure("TButton", background="#4a4a4a", foreground="white")
        style.map("TButton", background=[("active", "#5a5a5a")])
        style.configure("TEntry", background="#3c3c3c", foreground="white", fieldbackground="#3c3c3c")

        # Instructions
        tk.Label(root, text="Enter stock ticker and shares:", bg="#2c2c2c", fg="white").pack(pady=20)

        # Ticker input
        tk.Label(root, text="Ticker:", bg="#2c2c2c", fg="white").pack()
        self.ticker_entry = tk.Entry(root, width=20, bg="#3c3c3c", fg="white")
        self.ticker_entry.pack(pady=5)

        # Shares input
        tk.Label(root, text="Number of Shares:", bg="#2c2c2c", fg="white").pack()
        self.shares_entry = tk.Entry(root, width=20, bg="#3c3c3c", fg="white")
        self.shares_entry.pack(pady=5)

        # Run button
        ttk.Button(root, text="Calculate Risk", command=self.run_analysis).pack(pady=20)

        # Result label
        self.result_label = tk.Label(root, text="", bg="#2c2c2c", fg="white", font=("Arial", 12))
        self.result_label.pack(pady=20)

        # Plot canvas
        self.fig, self.ax = plt.subplots(figsize=(11, 7))  # Slightly larger figure size
        self.ax.set_facecolor("#2c2c2c")
        self.ax.tick_params(colors="white", labelsize=12)
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.fig.set_facecolor("#2c2c2c")
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(pady=20)

    def run_analysis(self):
        try:
            ticker = self.ticker_entry.get().strip().upper()
            shares = float(self.shares_entry.get().strip())

            if not ticker or shares <= 0:
                raise ValueError("Please enter a valid ticker and positive number of shares.")

            tickers = [ticker]
            shares_list = [shares]

            total_shares = sum(shares_list)
            weights = np.array([s / total_shares for s in shares_list])
            prices = get_stock_data(tickers)
            returns = calculate_returns(prices)
            pred_vol = predict_volatility(returns, weights)
            portfolio_var = calculate_var(returns, weights)
            percentage_move = predict_percentage_move(returns, weights)

            self.result_label.config(
                text=f"Predicted Volatility: {pred_vol:.4f}\n30-day VaR at 95%: {portfolio_var:.4f}\n"
                     f"Predicted 30-day % Move: {percentage_move:.2f}%"
            )

            # Calculate rolling volatility with a larger window for smoother results
            volatility = returns.rolling(window=60).std() * weights
            portfolio_volatility = volatility.sum(axis=1).dropna()

            # Ensure index is a proper datetime
            if not pd.api.types.is_datetime64_any_dtype(portfolio_volatility.index):
                portfolio_volatility.index = pd.to_datetime(portfolio_volatility.index)

            self.ax.clear()
            self.ax.plot(portfolio_volatility.index, portfolio_volatility, label="Historical Volatility", color="white", linewidth=2.0)
            self.ax.axhline(y=pred_vol, color="green", linestyle="--", label=f"Predicted Volatility ({pred_vol:.4f})")
            self.ax.axhline(y=portfolio_var, color="red", linestyle="--", label=f"30-day VaR ({portfolio_var:.4f})")
            self.ax.set_facecolor("#2c2c2c")
            self.ax.tick_params(colors="white", labelsize=12)
            self.ax.spines['bottom'].set_color('white')
            self.ax.spines['top'].set_color('white')
            self.ax.spines['right'].set_color('white')
            self.ax.spines['left'].set_color('white')
            self.ax.set_title("Portfolio Risk Over Time", color="white", fontsize=14, pad=15)
            self.ax.set_xlabel("Date", color="white", fontsize=12)
            self.ax.set_ylabel("Volatility / VaR", color="white", fontsize=12)

            # Format X-axis dates correctly with tighter control
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            self.ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show every 3 months for clarity
            plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45, ha='right', color='white', fontsize=10)

            # Adjust Y-axis limits for better visibility of small values
            y_min = min(portfolio_volatility.min(), portfolio_var, pred_vol) * 1.2
            y_max = max(portfolio_volatility.max(), portfolio_var, pred_vol) * 1.2
            self.ax.set_ylim(bottom=y_min, top=y_max)

            # Add subtle grid for better readability
            self.ax.grid(True, linestyle='--', alpha=0.3, color='white')

            # Position legend outside the plot for clarity, adjust size and padding
            self.ax.legend(facecolor="#2c2c2c", edgecolor="white", labelcolor="white",
                          loc="upper left", bbox_to_anchor=(1, 1), fontsize=10, borderpad=0.5, labelspacing=0.5)

            self.fig.set_facecolor("#2c2c2c")
            self.fig.tight_layout(pad=2.0)  # Adjust layout with more padding to prevent overlap
            self.fig.set_size_inches(11, 7)  # Slightly larger figure size for clarity
            self.canvas.draw()

        except Exception as e:
            # Custom error dialog with larger size
            error_window = tk.Toplevel(self.root)
            error_window.title("Error")
            error_window.geometry("400x200")  # Larger window size
            error_window.configure(bg="#2c2c2c")  # Match dark mode

            # Error message label
            tk.Label(error_window, text=str(e), bg="#2c2c2c", fg="white",
                     wraplength=350, justify="left", font=("Arial", 10)).pack(pady=20, padx=20)

            # OK button
            ttk.Button(error_window, text="OK", command=error_window.destroy,
                       style="TButton").pack(pady=10)

            # Center the window
            error_window.transient(self.root)
            error_window.grab_set()

if __name__ == "__main__":
    root = tk.Tk()
    app = PortfolioRiskApp(root)
    root.mainloop()