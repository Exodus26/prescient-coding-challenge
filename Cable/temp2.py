import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to identify optimal buy/sell points based on cumulative changes for maximum profit
def peak_to_trough_trades(changes, threshold=0.05):  # Threshold as percentage of cumulative change
    buy_dates = []
    sell_dates = []
    buy_changes = []
    sell_changes = []
    profits = []
    
    n = len(changes)
    cumulative_changes = changes.cumsum()  # Track the cumulative change over time

    # Initialize tracking variables
    min_change = cumulative_changes[0]  # The lowest change so far (buy point)
    min_date = changes.index[0]
    
    for i in range(1, n):
        current_change = cumulative_changes[i]
        current_date = changes.index[i]
        
        # Check if current_change is greater than min_change
        if current_change < min_change:
            # We found a new minimum change, update buy point
            min_change = current_change
            min_date = current_date
        
        # Calculate potential profit if we sell at current_change
        potential_profit = current_change - min_change
        
        # Check if we are at a peak (current change is less than previous change)
        is_peak = i > 1 and current_change < cumulative_changes[i - 1]

        # Check if the potential profit exceeds the threshold and we are at a peak
        if potential_profit >= threshold and is_peak:
            # Record the trade
            buy_dates.append(min_date)
            sell_dates.append(current_date)
            buy_changes.append(min_change)
            sell_changes.append(current_change)
            profits.append(potential_profit)
            
            # Reset min_change for the next potential trade
            min_change = current_change  # Reset the buy point after a successful sell
            min_date = current_date  # Reset to the current date as the new min date

    return pd.DataFrame({
        'Buy Date': buy_dates,
        'Buy Change': buy_changes,
        'Sell Date': sell_dates,
        'Sell Change': sell_changes,
        'Profit': profits
    })

# Load data (same as your current setup)
current_directory = os.getcwd()
current_directory = os.path.join(current_directory, 'Cable')
relative_path = os.path.join(current_directory, 'returns_train.csv')
data = pd.read_csv(relative_path)

# Convert 'month_end' to datetime and set as index
data['Dates'] = pd.to_datetime(data['month_end'])
data.set_index('Dates', inplace=True)

# Extract stock data (e.g., 'Stock1') as a series
changes1 = data['Stock1']

# Set a threshold (e.g., 5% change)
threshold_value = 0.25  # Threshold can be adjusted to your desired level

# Identify the optimal buy/sell trades based on cumulative changes and the threshold
trade_data = peak_to_trough_trades(changes1, threshold=threshold_value)

# Display the trade data
print(trade_data)

# Plot the stock changes and buy/sell points
plt.figure(figsize=(12, 6))
plt.plot(data.index, changes1.cumsum(), label='Cumulative Stock1 Changes')
plt.scatter(trade_data['Buy Date'], trade_data['Buy Date'].map(lambda x: changes1.cumsum()[x]), color='green', label='Buy', marker='^', s=100)
plt.scatter(trade_data['Sell Date'], trade_data['Sell Date'].map(lambda x: changes1.cumsum()[x]), color='red', label='Sell', marker='v', s=100)
plt.title('Peak-to-Trough Buy/Sell Points for Stock1 Based on Changes with Threshold')
plt.xlabel('Dates')
plt.ylabel('Cumulative Changes')
plt.legend()
plt.grid()
plt.show()

# Calculate total profit from all trades
total_profit = trade_data['Profit'].sum()
print(f"Total Profit: {total_profit}")
