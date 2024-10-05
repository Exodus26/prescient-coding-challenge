import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to identify optimal buy/sell points based on cumulative changes for maximum profit
import pandas as pd

# Function to identify optimal buy/sell points based on cumulative changes for maximum profit
def peak_to_trough_trades(changes, threshold=0.05):
    # Drop NaN values from changes
    changes = changes.dropna()

    # Track the cumulative change over time
    cumulative_changes = changes.cumsum()

    # Check if there are valid changes after dropping NaNs
    if cumulative_changes.empty:
        print("No valid data to analyze.")
        return pd.DataFrame(), pd.Series(dtype=bool)

    # Initialize tracking variables
    first_non_zero_index = cumulative_changes[cumulative_changes != 0].index[0]  # First non-zero index
    min_change = cumulative_changes.loc[first_non_zero_index]  # The lowest change so far (buy point)
    min_date = first_non_zero_index

    buy_dates = []
    sell_dates = []
    buy_changes = []
    sell_changes = []
    profits = []

    n = len(cumulative_changes)

    for i in range(1, n):
        current_change = cumulative_changes.iloc[i]
        current_date = cumulative_changes.index[i]
        
        # Check if current_change is less than min_change
        if current_change < min_change:
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

    # Create the trade DataFrame
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

def main():
    current_dir = os.path.join(os.getcwd(), 'Cable/2024 challenge')
    data_dir = os.path.join(current_dir, 'data')

    df_prices = pd.read_csv(os.path.join(data_dir, 'data1.csv'))

    df_prices['date'] = pd.to_datetime(df_prices['date'])
    df_prices.set_index('date', inplace=True)

    df_pivoted_prices = df_prices.pivot(columns='security', values='price')

    # Normalize each column based on the first non-zero entry
    normalized_prices = df_pivoted_prices.copy()

    df_price_diffs = normalized_prices.diff()

    for column in normalized_prices.columns:
        first_non_zero = normalized_prices[column][normalized_prices[column] != 0].iloc[0]
        normalized_prices[column] = normalized_prices[column] / first_non_zero

    # Create a DataFrame to hold the top trades indicators
    top_trades_df = pd.DataFrame(0, index=df_price_diffs.index, columns=df_price_diffs.columns)

    # Loop through each row of the normalized difference DataFrame
    df_price_diffs = normalized_prices.diff()

    # Create a DataFrame to hold the top trades indicators
    top_trades_df = pd.DataFrame(0, index=df_price_diffs.index, columns=df_price_diffs.columns)

    # Loop through each row of the normalized difference DataFrame
    for date, daily_changes in df_price_diffs.iterrows():
        # Get the top 10 highest positive changes for the day
        top_positive_changes = daily_changes[daily_changes > 0].nlargest(10)

        # If there are positive changes, mark them in the top trades DataFrame
        if not top_positive_changes.empty:
            top_trades_df.loc[date, top_positive_changes.index] = 1

    # Display the resulting DataFrame
    print("Top Trades DataFrame (1 = Top 10 positive trades, 0 = Not a top trade):")
    print(top_trades_df)

    # Optionally, save the DataFrame to a CSV file
    top_trades_df.to_csv(os.path.join(data_dir, 'top_trades.csv'))

if __name__ == '__main__':
    main()