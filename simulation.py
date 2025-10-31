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

stockTickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'DIS'] # Apple, Microsoft, Google, Amazon, Tesla, Meta, Nvidia, JPMorgan, Visa, Disney
endDate = dt.datetime.now() # today's date
startDate = endDate - dt.timedelta(days=300) # 300 days ago

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

# Summary statistics
ending_values = portfolio_sims[-1, :]

# Ending value percentiles
p5, p50, p95 = np.percentile(ending_values, [5, 50, 95])

# VaR/CVaR at 95% (loss-based)
losses = np.maximum(0.0, initial_investment - ending_values)
var95 = np.percentile(losses, 95)
tail_losses = losses[losses >= var95]
cvar95 = tail_losses.mean() if tail_losses.size > 0 else 0.0

# Max drawdown distribution
running_max = np.maximum.accumulate(portfolio_sims, axis=0)
drawdowns = (portfolio_sims - running_max) / running_max
max_drawdowns = drawdowns.min(axis=0) # negative numbers
md_p5, md_p50, md_p95 = np.percentile(-max_drawdowns, [5, 50, 95])

# Threshold probabilities
lower_threshold = 0.9 * initial_investment
upper_threshold = 1.2 * initial_investment
prob_below_lower = np.mean(ending_values < lower_threshold)
prob_above_upper = np.mean(ending_values > upper_threshold)

print("\nSummary statistics ({} sims, {} days):".format(num_simulations, num_days))
print("Ending value percentiles (USD): p5={:,.0f}, p50={:,.0f}, p95={:,.0f}".format(p5, p50, p95))
print("VaR95 (USD loss): {:,.0f}".format(var95))
print("CVaR95 (USD loss): {:,.0f}".format(cvar95))
print("Max drawdown percentiles (%): p5={:.1f}, p50={:.1f}, p95={:.1f}".format(100*md_p5, 100*md_p50, 100*md_p95))
print("P(ending < {:,.0f}) = {:.1%}".format(lower_threshold, prob_below_lower))
print("P(ending > {:,.0f}) = {:.1%}".format(upper_threshold, prob_above_upper))

# Plot the results
plt.plot(portfolio_sims)
plt.title('Monte Carlo Simulation of Portfolio Value')
plt.xlabel('Days')
plt.ylabel('Portfolio Value (USD)')
plt.show()

