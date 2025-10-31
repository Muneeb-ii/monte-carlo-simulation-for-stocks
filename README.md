# Monte Carlo Portfolio Simulation

## Overview
A simple Monte Carlo engine that:
- pulls historical prices from Yahoo Finance,
- estimates average returns and correlations,
- simulates many plausible future paths for a portfolio,
- and summarizes risk/return (percentiles, VaR/CVaR, drawdowns).

## Installation and Setup
```bash
# 1) Clone or cd into the project
git clone https://github.com/Muneeb-ii/monte-carlo-simulation-for-stocks
cd monte-carlo-simulation-for-stocks

# 2) Create and activate a virtual environment (macOS/Linux)
python3 -m venv .venv
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Run the simulation
python3 simulation.py
```

If you update dependencies later:
```bash
pip freeze > requirements.txt
```

## Configuration
- `stockTickers`: list of tickers to include. Order matters and must match your weights.
- `lookback_days` or `startDate`/`endDate`: how much past data to learn from. Longer = more stable, shorter = more responsive.
- `weights`: fraction of portfolio in each asset. Must sum to 1. Controls how asset moves blend into portfolio moves.
- `num_simulations`: how many future paths to simulate. More = smoother stats (slower to run).
- `num_days`: horizon length in trading days (e.g., 252 ≈ 1 year).
- `initial_investment`: starting portfolio value in USD.
- `lower_threshold`, `upper_threshold`: targets for probability checks in the summary.
- Confidence level (conceptual): VaR/CVaR is at 95% in the code; you can change this by editing how the percentile is computed.

## Methodology

### Data and Estimation
- Download adjusted close prices with `yfinance`.
- Compute daily simple returns and then:
  - Mean vector $\mu$ = average daily return per asset.
  - Covariance matrix $\Sigma$ = how assets vary and move together.

### Simulation Model
- **Non-math**: Each day, we “roll the dice” for each stock so that the random moves have the right typical size and co-movement. We blend stock moves using your weights to get the portfolio move and update the portfolio value. Repeat for many paths to see a distribution of outcomes.

- **Math**:
  - Assume daily asset returns follow a multivariate normal:
    $$
    r_t \sim \mathcal{N}(\mu, \Sigma)
    $$
  - Draw independent standard normals: $z_t \sim \mathcal{N}(0, I)$.
  - Cholesky factorization: $\Sigma = L L^\top$.
  - Create correlated draws:
    $$
    r_t = \mu + L z_t
    $$
  - Portfolio return that day:
    $$
    r^{(p)}_t = w^\top r_t
    $$
  - Portfolio value evolves multiplicatively:
    $$
    V_t = V_{t-1} \times (1 + r^{(p)}_t)
    $$

### Summary Statistics
- **Ending value percentiles (p5, p50, p95)**: distribution of final portfolio value.
- **VaR95 (loss)**: the 95th percentile of loss at horizon:
  $$
  \mathrm{VaR}_{95} = \mathrm{percentile}_{95}(\max(0, V_0 - V_T))
  $$
- **CVaR95 (loss)**: average loss given loss ≥ VaR95 (tail average).
- **Max drawdown distribution**: worst peak-to-trough fall during the path. Reported as percentiles across simulations.
- **Target probabilities**: P(ending below floor) and P(ending above goal).

## Interpreting Results

### Plot
- **Left panel — Simulated paths with percentile bands**:
  - Light blue lines are individual simulated portfolio paths.
  - A bold black line shows the median path (p50).
  - A shaded band shows the 5–95% percentile range across days.
  - Dashed horizontal reference lines mark the ending p5, p50, and p95 levels.
- **Right panel — Ending value distribution**:
  - Histogram of ending portfolio values across all simulations.
  - Dashed vertical lines mark p5, p50, and p95 of ending values.
  - Wider spread implies higher outcome uncertainty; left tail highlights downside risk.

### Summary Output (Example)
- Ending value percentiles (USD): p5=89,500, p50=105,800, p95=124,200
  - Typical end value ≈ 105.8k; only 5% of runs end below 89.5k or above 124.2k.
- VaR95 (USD loss): 9,800
  - There’s a 5% chance the end-of-horizon loss exceeds 9.8k relative to the initial 100k.
- CVaR95 (USD loss): 12,700
  - On average, in the worst 5% of runs, loss is about 12.7k.
- Max drawdown percentiles (%): p5=12.3, p50=18.9, p95=28.7
  - A typical worst drop from a prior peak is ~19%; severe-but-plausible is ~29%.
- P(ending < 90,000) = 7.4%
  - Around 7.4% chance to finish below 90k.
- P(ending > 120,000) = 22.1%
  - Around 22.1% chance to finish above 120k.

### How to Use These
- Compare allocations by their downside (VaR/CVaR, p5) vs upside (p95).
- Check if drawdowns exceed your risk tolerance.
- Set targets and see the probability of hitting or missing them.
- Adjust `num_simulations` and `num_days` to your horizon and desired precision.

## Limitations and Extensions
- Historical estimates may not hold in the future (regime changes, crises).
- Normality understates tails; enhancements include fat-tailed distributions, time-varying volatility (e.g., GARCH), copulas, or bootstrapping returns.
- Cholesky can fail if $\Sigma$ isn’t positive definite; regularization or eigen-based transforms can be used in such cases.

## Data Source
- Prices via Yahoo Finance using `yfinance`. Availability/quality depends on Yahoo’s data.
