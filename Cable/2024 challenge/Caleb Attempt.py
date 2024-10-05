import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# Function to identify optimal buy/sell points based on cumulative changes for maximum profit
import pandas as pd

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

    df_training = top_trades_df.loc[:'2024-01-01']
    print(df_training)

if __name__ == '__main__':
    main()