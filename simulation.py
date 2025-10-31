import pandas as pd
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

portfolio_sims = np.full(shape=(num_days, num_simulations), fill_value=0.0) # create a matrix of zeros for the number of simulations and the number of days

initial_investment = 100000 # initial investment

# We will be assuming that daily returns are distributed by a Multivariate Normal Distribution

for simulation in range(num_simulations):
    Z = np.random.normal(size=(num_days, len(weights))) # generate random numbers for the number of days and the number of assets from a standard normal distribution 
    L = np.linalg.cholesky(covMatrix) # Cholesky Decomposition of the covariance matrix: a lower triangular matrix such that the covariance matrix is equal to the product of the transpose of the lower triangular matrix and the lower triangular matrix
    dailyReturns = meanM + np.inner(L, Z) # daily returns are equal to the mean returns plus the product of the lower triangular matrix and the random numbers (the inner product of the two matrices)
    portfolio_sims[:,simulation] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initial_investment # cumulative product of the daily returns plus 1 (to account for the initial investment) multiplied by the initial investment

plt.plot(portfolio_sims)
plt.title('Monte Carlo Simulation of Portfolio Value')
plt.xlabel('Days')
plt.ylabel('Portfolio Value (USD)')
plt.show()