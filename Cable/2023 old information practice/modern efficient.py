import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic stock return data
np.random.seed(42)  # For reproducibility
n_assets = 40
n_months = 8 * 12  # 8 years of monthly data
returns = np.random.randn(n_months, n_assets)  # Normally distributed returns

# Create a DataFrame for better visualization
stocks = [f'Stock {i+1}' for i in range(n_assets)]
returns_df = pd.DataFrame(returns, columns=stocks)

# Calculate expected returns and covariance matrix
expected_returns = returns_df.mean()
cov_matrix = returns_df.cov()

# Mean-Variance Optimization
def optimize_portfolio(expected_returns, cov_matrix, num_portfolios=10000):
    results = np.zeros((num_portfolios, 3))  # [Returns, Risk (StdDev), Weights]
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(expected_returns))
        weights /= np.sum(weights)  # Normalize weights to sum to 1

        portfolio_return = np.dot(weights, expected_returns)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        results[i, 0] = portfolio_return
        results[i, 1] = portfolio_std_dev
        results[i, 2] = weights  # Store the entire weights array in the results

        weights_record.append(weights)

    return results, weights_record

# Run optimization
results, weights_record = optimize_portfolio(expected_returns, cov_matrix)

# Create a DataFrame for results
results_df = pd.DataFrame(results, columns=['Return', 'Risk', 'Weights'])
results_df['Sharpe Ratio'] = results_df['Return'] / results_df['Risk']

# Plotting the Efficient Frontier
plt.figure(figsize=(10, 7))
plt.scatter(results_df['Risk'], results_df['Return'], c=results_df['Sharpe Ratio'], cmap='viridis', marker='o')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Portfolio Risk (Standard Deviation)')
plt.ylabel('Portfolio Return')
plt.title('Markowitz Efficient Frontier')
plt.grid()

# Highlight the maximum Sharpe Ratio portfolio
max_sharpe_idx = results_df['Sharpe Ratio'].idxmax()
sdp = results_df.iloc[max_sharpe_idx]
plt.scatter(sdp['Risk'], sdp['Return'], color='red', marker='*', s=200, label='Max Sharpe Ratio Portfolio')

# Show the plot
plt.legend()
plt.show()

# Display the optimal weights for the maximum Sharpe ratio portfolio
optimal_weights = weights_record[max_sharpe_idx]
optimal_portfolio = pd.Series(optimal_weights, index=stocks)
print("Optimal Portfolio Weights for Maximum Sharpe Ratio:")
print(optimal_portfolio)
