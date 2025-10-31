import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

# Import data
def get_data(tickers, start_date, end_date):
    stockData = yf.download(tickers, start=start_date, end=end_date, progress=False, threads=True)
    stockData = stockData['Close']
    returns = stockData.pct_change(fill_method=None)
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

stockTickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'DIS']
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stockTickers, startDate, endDate)

weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

# Monte Carlo Simulation
num_simulations = 100
num_days = 100

# Create a numeric matrix of mean returns with shape (num_assets, num_days)
mean_vec = meanReturns.values
meanM = np.repeat(mean_vec[:, None], num_days, axis=1)

portfolio_sims = np.full(shape=(num_days, num_simulations), fill_value=0.0) # matrix of zeros with shape (num_days, num_simulations)

initial_investment = 100000 # initial investment

# Assume daily asset returns follow a multivariate normal distribution

for simulation in range(num_simulations):
    Z = np.random.normal(size=(num_days, len(weights))) # standard normal shocks for each day and asset
    L = np.linalg.cholesky(covMatrix) # Cholesky factor L of covariance where cov = L @ L.T (L is lower-triangular)
    dailyReturns = meanM + np.inner(L, Z) # mean matrix plus correlated shocks via Cholesky (shape: assets x days)
    portfolio_sims[:,simulation] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initial_investment # cumulative product of the daily returns plus 1 (to account for the initial investment) multiplied by the initial investment

plt.plot(portfolio_sims)
plt.title('Monte Carlo Simulation of Portfolio Value')
plt.xlabel('Days')
plt.ylabel('Portfolio Value (USD)')
plt.show()