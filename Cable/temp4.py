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
relative_path = os.path.join(current_directory, 'returns_train.csv')
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

# Display the updated data DataFrame
# print(data)

monthly_profits = pd.DataFrame(index=data.index)

# Calculate monthly profits for each stock and sum them
for column in data.columns:
    if 'Profit' in column:
        monthly_profits[column] = data[column]

print(monthly_profits)


# Create a new DataFrame for averaged profits
averaged_profits = pd.DataFrame(index=monthly_profits.index)

# Divide each profit value by the number of times it appears in succession
for column in monthly_profits.columns:
    profits = monthly_profits[column]
    constant_segments = (profits != profits.shift()).cumsum()  # Identify segments where the value changes

    # Count occurrences in each segment
    counts = profits.groupby(constant_segments).transform('size')

    # Divide the profit by the count for each segment
    divided_column = profits / counts

    # Store the divided profits in the new DataFrame
    averaged_profits[column] = divided_column

# Display the new DataFrame containing averaged profits
print(averaged_profits)

# Create a new DataFrame for adjusted values
new_adjusted_profits = pd.DataFrame(index=averaged_profits.index, columns=averaged_profits.columns)

# Iterate through each row of the averaged profits DataFrame
for index, row in averaged_profits.iterrows():
    row_values = row.copy()
    adjusted_row = pd.Series(0, index=row.index)  # Initialize the adjusted row with zeros
    
    # Get the top 10 indices based on values
    top_indices = row_values.nlargest(10).index.tolist()
    
    # If there are fewer than 10 items greater than 0, adjust accordingly
    count_greater_than_zero = (row_values > 0).sum()

    # Add 0.1 for each of the top values (or as many as exist)
    for idx in top_indices:
        adjusted_row[idx] = 0.1

    # If there are fewer than 10 items greater than 0
    if count_greater_than_zero < 10:
        # Randomly add 0.01 to the remaining entries until the sum equals 1
        while adjusted_row.sum() < 1:
            random_index = np.random.choice(row_values.index[row_values > 0], size=1)[0]  # Pick a random index of a non-zero value
            if adjusted_row[random_index] < 0.1:  # Ensure we only add to valid positions
                adjusted_row[random_index] += 0.01

    # Normalize if the sum is still less than 1
    while adjusted_row.sum() < 1:
        for col in adjusted_row.index:
            if adjusted_row.sum() < 1:
                adjusted_row[col] += 0.01

    # Store the adjusted row in the new DataFrame
    new_adjusted_profits.loc[index] = adjusted_row

# Display the new DataFrame containing adjusted profits
print(new_adjusted_profits)

for index, row in new_adjusted_profits.iterrows():
    print(f"Sum {row.sum()}")
    print(f"Max {row.max()}")
