# %%
# BROOM???
# OH NO A MOP!!!
import numpy as np
import pandas as pd
import datetime
import os

import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

print('---> Python Script Start', t0 := datetime.datetime.now())

# %%

print('---> the parameters')

# training and test dates

start_train = datetime.date(2017, 1, 1)
end_train = datetime.date(2023, 12, 30) # gap for embargo (no overlap between train and test)
start_test = datetime.date(2024, 1, 1) # test set is this datasets 2024 data
end_test = datetime.date(2024, 6, 30)

case = 'TEST'   #change as required

verbose = False

print('---> initial data set up')

# sector data

# Set up directories
current_dir = os.path.join(os.getcwd(), 'Sam/2024 challenge')
data_dir = os.path.join(current_dir, 'data')

# Load sector data
df_sectors = pd.read_csv(os.path.join(data_dir, 'data0.csv'))

# Load price data
df_data = pd.read_csv(os.path.join(data_dir, 'data1.csv'))
df_data['date'] = pd.to_datetime(df_data['date'])
df_data.set_index('date', inplace=True)
df_pivoted_prices = df_data.pivot(columns='security', values='price')

# Load returns data
df_returns = pd.read_csv(os.path.join(data_dir, 'returns.csv'))
df_returns['date'] = pd.to_datetime(df_returns['date'])
df_returns = df_returns.pivot(index='date', columns='security', values='return1')

# Calculate price differences and ensure index is set correctly
df_price_diffs = df_pivoted_prices.diff()
# We don't need to convert df_price_diffs['date'] to datetime since it does not have a 'date' column
df_price_diffs = df_price_diffs.fillna(0)  # Fill NaN values with 0 before saving to CSV


# Slicing for training and testing datasets
# Ensure to slice using the index directly
df_train_prices = df_pivoted_prices['2017-01-31':'2023-12-31']
df_test_prices = df_pivoted_prices['2024-01-01':'2024-06-30']
df_train_returns = df_returns['2017-01-31':'2023-12-31']  # Ensure dates align with the train prices
df_test_returns = df_returns['2024-01-01':'2024-06-30']
df_train_price_diffs = df_price_diffs['2017-01-31':'2023-12-31']
df_test_price_diffs = df_price_diffs['2024-01-01':'2024-06-30']


if case == 'TEST':
    df_prices = df_test_prices
    df_returns = df_test_returns
    df_price_diffs = df_test_price_diffs
elif case == 'TRAIN':
    df_prices = df_train_prices
    df_returns = df_train_returns
    df_price_diffs = df_train_price_diffs


#SIMPLE MOVING AVERAGE SIGNAL STRENGTH APPROACH
# Calculate the 50-day moving average for each stock in df_pivoted_prices
df_50_day_moving_avg = df_pivoted_prices.rolling(window=50).mean()
# print('df_50_day_moving_avg\n', df_50_day_moving_avg)

# Calculate the 10-day moving average for each stock in df_pivoted_prices
df_10_day_moving_avg = df_pivoted_prices.rolling(window=10).mean()
# print('df_10_day_moving_avg\n', df_10_day_moving_avg)

# Calculate signal strength by comparing simple moving averages:
df_signal_strength_sma = (df_10_day_moving_avg - df_50_day_moving_avg) / df_50_day_moving_avg
# print('df_signal_strength\n', df_signal_strength_sma)

df_top_trades_sma = pd.DataFrame(0, index=df_price_diffs.index, columns=df_price_diffs.columns)
# Loop through each row of the normalized difference DataFrame
for date, df_signal_strength_sma in df_price_diffs.iterrows():
    # Get the top 10 highest positive changes for the day
    top_positive_changes = df_signal_strength_sma[df_signal_strength_sma >= 0].nlargest(10)
    # If there are positive changes, mark them in the top trades DataFrame
    if not top_positive_changes.empty:
        df_top_trades_sma.loc[date, top_positive_changes.index] = 1

df_top_trades_sma.to_csv(os.path.join(data_dir, 'top_trades.csv'))

# Display the resulting DataFrame
print("Top Trades DataFrame SMA (1 = Top 10 positive trades, 0 = Not a top trade):")
print(df_top_trades_sma)


# Calculate the returns based on buy signals
df_total_returns2 = (df_top_trades_sma/10) * df_price_diffs
# Sum the returns for each stock
total_return_per_stock2 = df_total_returns2.sum()
# Calculate the total return across all stocks
total_return_all_stocks2 = df_total_returns2.sum().sum()  # Sum all the values in the DataFrame
print("Total return for each stock SMA:")
print(total_return_per_stock2)
print("\nTotal return for all stocks combined SMA:")
print(total_return_all_stocks2)

def plot_payoff(df_buys):

    df = df_buys.copy()

    # assert (df.sum(axis=1)==10).sum() == len(df), '---> must have exactly 10 buys each day'

    # matrix of buys
    df_payoff = df[['date']].copy()
    del df['date']
    arr_buys = np.array(df)
    arr_buys = arr_buys*(1/10) # equally weighted

    # return matrix
    arr_ret = np.array(df_returns)
    arr_ret = arr_ret + 1

    df_payoff['payoff'] = (arr_buys * arr_ret @ np.ones(len(df_sectors)).reshape((len(df_sectors), 1)))[:, 0]
    df_payoff['tri'] = df_payoff['payoff'].cumprod()

    fig_payoff = px.line(df_payoff, x='date', y='tri')
    fig_payoff.show()

    print(f"---> payoff for these buys between period {df_payoff['date'].min()} and {df_payoff['date'].max()} is {(df_payoff['tri'].values[-1]-1)*100 :.2f}%")

    return df_payoff

df_payoff = plot_payoff(df_top_trades_sma)


# %%

print('---> Python Script End', t1 := datetime.datetime.now())
print('---> Total time taken', t1 - t0)


