# Stock Portfolio Risk Predictor

A Python-based graphical tool to analyze and predict the risk of a stock portfolio using machine learning. This open-source project provides a dark-mode interface to visualize historical volatility, predict future volatility, calculate Value-at-Risk (VaR), and estimate the percentage move over 30 days.

## Features
- Input a stock ticker and number of shares to analyze.
- Visualize historical volatility over time.
- Predict future volatility using an LSTM neural network.
- Calculate 30-day Value-at-Risk (VaR) at 95% confidence.
- Estimate the predicted percentage move in portfolio value over 30 days.
- Dark-mode, user-friendly GUI with clear visualizations.

## Installation

### 1. Clone the repository:
   bash:

   git clone https://github.com/bgilbert0112/StockPortfolioRiskPredictor.git
   cd StockPortfolioRiskPredictor
   
### 2. Create a virtual environment (optional but recommended):

bash:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### 3. Install the required dependencies:

bash:
pip install -r requirements.txt
Run the application:
bash
python portfolio_risk.py

### 4. Usage

Launch the application by running python portfolio_risk.py.
Enter a stock ticker (e.g., TSLA) in the "Ticker" field.
Enter the number of shares in the "Number of Shares" field.
Click "Calculate Risk" to see the results and visualization.
The result label will display:
Predicted Volatility
30-day VaR at 95%
Predicted 30-day % Move
The graph will show the portfolio risk over time, including historical volatility, predicted volatility, and VaR.
Contributing
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

### 5. Support the Developer

Love this program? Want to buy me a coffee to support further development? Your support is greatly appreciated! You can send a donation via Bitcoin to the following wallet address:

Bitcoin Wallet: bc1qwgpr8keenl8zggnt2823wpkjmjky5ga8pzuvkd

Thank you for your support!