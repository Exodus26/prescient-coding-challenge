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
end_train = datetime.date(2023, 11, 30) # gap for embargo (no overlap between train and test)
start_test = datetime.date(2024, 1, 1) # test set is this datasets 2024 data
end_test = datetime.date(2024, 6, 30)

n_buys = 10
verbose = False

print('---> initial data set up')

# sector data

current_dir = os.path.join(os.getcwd(),'Sam/2024 challenge')
data_dir = os.path.join(current_dir, 'data')

df_sectors = pd.read_csv(os.path.join(data_dir,'data0.csv'))
df_data = pd.read_csv(os.path.join(data_dir,'data1.csv'))
df_data['date'] = pd.to_datetime(df_data['date']).apply(lambda d: d.date())
df_data['date'] = pd.to_datetime(df_data['date'])
df_data.set_index('date', inplace=True)

df_pivoted_prices = df_data.pivot(columns='security', values='price')
# print('df_pivoted_prices\n', df_pivoted_prices)

df_price_diffs = df_pivoted_prices.diff()
# print('df_price_diffs\n', df_price_diffs)

#SIMPLE MOVING AVERAGE SIGNAL STRENGTH APPROACH

# Calculate the 50-day moving average for each stock in df_pivoted_prices
df_50_day_moving_avg = df_pivoted_prices.rolling(window=50).mean()
print('df_50_day_moving_avg\n', df_50_day_moving_avg)

# Calculate the 10-day moving average for each stock in df_pivoted_prices
df_10_day_moving_avg = df_pivoted_prices.rolling(window=10).mean()
print('df_10_day_moving_avg\n', df_10_day_moving_avg)

# Calculate signal strength by comparing simple moving averages:
df_signal_strength = (df_10_day_moving_avg - df_50_day_moving_avg) / df_50_day_moving_avg
print('df_signal_strength\n', df_signal_strength)

#EXPONENTIAL MOVING AVERAGE SIGNAL STRENGTH APPROACH
# Custom weighting factor (between 0 and 1)
custom_weight = 2  # Adjust this value as needed

# Calculate the 50-day Weighted Exponential Moving Average (WEMA)
def weighted_ema(prices, span, custom_weight):
    # Create an array of weights with float type
    weights = np.array([(1 - custom_weight) * (custom_weight ** i) for i in range(span)], dtype=float)
    weights /= weights.sum()  # Normalize weights
    return prices.rolling(window=span).apply(lambda x: np.dot(weights, x), raw=True)

# Calculate the 50-day Weighted EMA for each stock
df_50_day_wema = weighted_ema(df_pivoted_prices, span=50, custom_weight=custom_weight)
print('df_50_day_wema\n', df_50_day_wema)

# Calculate the 10-day Weighted EMA for each stock
df_10_day_wema = weighted_ema(df_pivoted_prices, span=10, custom_weight=custom_weight)
print('df_10_day_wema\n', df_10_day_wema)

# Calculate signal strength by comparing the Weighted Exponential Moving Averages:
df_signal_strength_wema = (df_10_day_wema - df_50_day_wema) / df_50_day_wema
print('df_signal_strength_wema\n', df_signal_strength_wema)

#MEAN RETURN - VARIANCE signal strength approach
df_mean_returns_per_day = df_price_diffs.sum(axis=1) / 100  #100 stocks, change as needed
print('Mean Returns per day:\n', df_mean_returns_per_day)

df_variances = pd.DataFrame(index=df_price_diffs.index, columns=df_price_diffs.columns)

for stock in df_price_diffs.columns:
    # Calculate the squared difference from the mean return
    squared_diff = (df_price_diffs[stock] - df_mean_returns_per_day) ** 2
    # Assign this to the variances DataFrame
    df_variances[stock] = squared_diff

df_stock_variances = df_variances.mean(axis=0)  # Mean variance for each stock
print('Variance for each stock:\n', df_stock_variances)

df_signal_strength = pd.DataFrame(index=df_price_diffs.index, columns=df_price_diffs.columns)

for stock in df_price_diffs.columns:
    # Calculate the signal strength using the mean return and the stock variance
    signal_strength = (df_price_diffs[stock] - df_mean_returns_per_day) / df_variances[stock]
    df_signal_strength[stock] = signal_strength

print('Signal Strength for all stocks:\n', df_signal_strength)
