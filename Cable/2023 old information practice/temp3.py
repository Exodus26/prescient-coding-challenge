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
    min_change = cumulative_changes.iloc[0]  # The lowest change so far (buy point)
    min_date = changes.index[0]
    
    for i in range(1, n):
        current_change = cumulative_changes.iloc[i]
        current_date = changes.index[i]
        
        # Check if current_change is greater than min_change
        if current_change < min_change:
            # We found a new minimum change, update buy point
            min_change = current_change
            min_date = current_date
        
        # Calculate potential profit if we sell at current_change
        potential_profit = current_change - min_change
        
        # Check if we are at a peak (current change is less than previous change)
        is_peak = i > 1 and current_change < cumulative_changes.iloc[i - 1]

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

    trade_df = pd.DataFrame({
        'Buy Date': buy_dates,
        'Buy Change': buy_changes,
        'Sell Date': sell_dates,
        'Sell Change': sell_changes,
        'Profit': profits
    })
    
    # Create a holding column
    all_dates = changes.index  # Get all dates from the original data
    holding_status = pd.Series(False, index=all_dates)  # Initialize the holding status column to False

    for index, row in trade_df.iterrows():
        # Mark the dates between buy and sell (inclusive)
        holding_status.loc[(holding_status.index >= row['Buy Date']) & (holding_status.index <= row['Sell Date'])] = True

    # Add the holding status to the trade DataFrame
    trade_df['Holding Status'] = holding_status[trade_df['Buy Date']].reindex(trade_df['Buy Date']).fillna(False).values
    
    return trade_df, holding_status  # Return both the trade DataFrame and holding status

# Load data (same as your current setup)
current_directory = os.getcwd()
current_directory = os.path.join(current_directory, 'Cable')
relative_path = os.path.join(current_directory, '2024 old information\\returns.csv')
data = pd.read_csv(relative_path)

# Convert 'month_end' to datetime and set as index
data['Dates'] = pd.to_datetime(data['month_end'])
data.set_index('Dates', inplace=True)

# Set a threshold (e.g., 5% change)
threshold_value = 0.25  # Threshold can be adjusted to your desired level

# Process each stock column
for column in data.columns:
    if 'Stock' in column:
        changes = data[column]

        # Identify the optimal buy/sell trades based on cumulative changes and the threshold
        trade_data, holding_status = peak_to_trough_trades(changes, threshold=threshold_value)  # Capture both outputs

        # Display the trade data for this stock
        print(f"Trade data for {column}:")
        print(trade_data)

        # Calculate total profit from all trades
        total_profit = trade_data['Profit'].sum()
        print(f"Total Profit for {column}: {total_profit}")

        # Add the Holding Status to the original data
        data[f'Holding Status {column}'] = holding_status

        # Create a new column for profit that reflects the profits during the holding period
        data[f'Profit {column}'] = np.nan  # Initialize profit column
        for index, row in trade_data.iterrows():
            data.loc[row['Buy Date']:row['Sell Date'], f'Profit {column}'] = row['Profit']

        # Optionally, fill NaN values in the Profit column with 0 (for dates without trades)
        data[f'Profit {column}'].fillna(0, inplace=True)

        # Plot the stock changes and buy/sell points
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, changes.cumsum(), label=f'Cumulative {column} Changes')
        plt.scatter(trade_data['Buy Date'], trade_data['Buy Change'], color='green', label='Buy', marker='^', s=100)
        plt.scatter(trade_data['Sell Date'], trade_data['Sell Change'], color='red', label='Sell', marker='v', s=100)
        plt.title(f'Peak-to-Trough Buy/Sell Points for {column} Based on Changes with Threshold')
        plt.xlabel('Dates')
        plt.ylabel('Cumulative Changes')
        plt.legend()
        plt.grid()
        plt.show()

# Display the updated data DataFrame
print(data)
