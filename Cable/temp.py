import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to identify optimal buy/sell points based on cumulative changes for maximum profit
def peak_to_trough_trades(changes):
    buy_dates = []
    sell_dates = []
    buy_changes = []
    sell_changes = []
    profits = []
    
    i = 0
    n = len(changes)
    
    cumulative_changes = changes.cumsum()  # Track the cumulative change over time
    
    while i < n - 1:
        # Step 1: Find the local minimum in cumulative changes (buy point)
        while i < n - 1 and cumulative_changes[i+1] <= cumulative_changes[i]:
            i += 1
        if i == n - 1:  # No more trading points
            break
        buy_change = cumulative_changes[i]
        buy_date = changes.index[i]
        
        # Step 2: Find the local maximum after the local minimum (sell point)
        while i < n - 1 and cumulative_changes[i+1] >= cumulative_changes[i]:
            i += 1
        sell_change = cumulative_changes[i]
        sell_date = changes.index[i]
        
        # Step 3: Calculate the profit as the difference in cumulative changes
        profit = sell_change - buy_change
        if profit > 0:
            buy_dates.append(buy_date)
            sell_dates.append(sell_date)
            buy_changes.append(buy_change)
            sell_changes.append(sell_change)
            profits.append(profit)
    
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

# Identify the optimal buy/sell trades based on cumulative changes
trade_data = peak_to_trough_trades(changes1)

# Display the trade data
print(trade_data)

# Plot the stock changes and buy/sell points
plt.figure(figsize=(12, 6))
plt.plot(data.index, changes1.cumsum(), label='Cumulative Stock1 Changes')
plt.scatter(trade_data['Buy Date'], trade_data['Buy Date'].map(lambda x: changes1.cumsum()[x]), color='green', label='Buy', marker='^', s=100)
plt.scatter(trade_data['Sell Date'], trade_data['Sell Date'].map(lambda x: changes1.cumsum()[x]), color='red', label='Sell', marker='v', s=100)
plt.title('Peak-to-Trough Buy/Sell Points for Stock1 Based on Changes')
plt.xlabel('Dates')
plt.ylabel('Cumulative Changes')
plt.legend()
plt.grid()
plt.show()

# Calculate total profit from all trades
total_profit = trade_data['Profit'].sum()
print(f"Total Profit: {total_profit}")
