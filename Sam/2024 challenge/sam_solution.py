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

case = 'TEST'   #change as required

verbose = False

print('---> initial data set up')

# sector data

current_dir = os.path.join(os.getcwd(),'Sam/2024 challenge')
data_dir = os.path.join(current_dir, 'data')

# df_sectors = pd.read_csv(os.path.join(data_dir,'data0.csv'))
df_data = pd.read_csv(os.path.join(data_dir,'data1.csv'))
df_data['date'] = pd.to_datetime(df_data['date']).apply(lambda d: d.date())
df_data['date'] = pd.to_datetime(df_data['date'])
df_data.set_index('date', inplace=True)
df_pivoted_prices = df_data.pivot(columns='security', values='price')
# print('df_pivoted_prices\n', df_pivoted_prices)

df_returns = pd.read_csv(os.path.join(data_dir,'returns.csv'))
df_returns['date']= pd.to_datetime(df_returns['date']).apply(lambda d: d.date())
df_returns = df_returns[df_returns['date']>=start_test]
df_returns = df_returns.pivot(index='date', columns='security', values='return1')
# print(df_returns)


df_price_diffs = df_pivoted_prices.diff()
# print('df_price_diffs\n', df_price_diffs)


# # Filtering prices and returns
# df_train_prices = df_pivoted_prices[(df_pivoted_prices.index >= start_train) & (df_pivoted_prices.index <= end_train)]
# df_test_prices = df_pivoted_prices[(df_pivoted_prices.index >= start_test) & (df_pivoted_prices.index <= end_test)]
# df_train_returns = df_returns[(df_returns.index >= start_train) & (df_returns.index <= end_train)]
# df_test_returns = df_returns[(df_returns.index >= start_test) & (df_returns.index <= end_test)]
# df_train_price_diffs = df_price_diffs[(df_price_diffs.index >= start_train) & (df_price_diffs.index <= end_train)]
# df_test_price_diffs = df_price_diffs[(df_price_diffs.index >= start_test) & (df_price_diffs.index <= end_test)]

# if case == 'TEST':
#     df_prices = df_test_prices
#     df_returns = df_test_returns
#     df_price_diffs = df_test_price_diffs
# elif case == 'TRAIN':
#     df_prices = df_train_prices
#     df_returns = df_train_returns
#     df_price_diffs = df_train_price_diffs


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
    top_positive_changes = df_signal_strength_sma[df_signal_strength_sma > 0].nlargest(10)
    # If there are positive changes, mark them in the top trades DataFrame
    if not top_positive_changes.empty:
        df_top_trades_sma.loc[date, top_positive_changes.index] = 1

# Display the resulting DataFrame
print("Top Trades DataFrame SMA (1 = Top 10 positive trades, 0 = Not a top trade):")
print(df_top_trades_sma)


# MEAN RETURN - VARIANCE signal strength approach
df_mean_returns_per_day = df_price_diffs.sum(axis=1) / 100  # 100 stocks, change as needed
# print('Mean Returns per day:\n', df_mean_returns_per_day)

# Calculate variances for each stock
df_variances = pd.DataFrame(index=df_price_diffs.index, columns=df_price_diffs.columns)

for stock in df_price_diffs.columns:
    # Calculate the variance for each stock (mean of squared differences)
    df_variances[stock] = (df_price_diffs[stock] - df_price_diffs[stock].mean()) ** 2

# Now calculate the mean variance for each stock
df_stock_variances = df_variances.mean(axis=0)  # Mean variance for each stock (columns)
# print('Variance for each stock:\n', df_stock_variances)

# Create an empty DataFrame for signal strength
df_signal_strength_mv = pd.DataFrame(index=df_price_diffs.index, columns=df_price_diffs.columns)

for stock in df_price_diffs.columns:
    # Calculate the variance for this stock
    stock_variance = df_stock_variances[stock]
    
    # Calculate the signal strength using the mean return and the stock variance
    if stock_variance != 0:  # Avoid division by zero
        signal_strength = (df_price_diffs[stock] - df_mean_returns_per_day) / stock_variance
    else:
        signal_strength = np.nan  # If variance is zero, set signal strength to NaN
    df_signal_strength_mv[stock] = signal_strength

# print('Signal Strength:\n', df_signal_strength_mv)

df_top_trades_mv = pd.DataFrame(0, index=df_price_diffs.index, columns=df_price_diffs.columns)
# Loop through each row of the normalized difference DataFrame
for date, df_signal_strength_mv in df_price_diffs.iterrows():
    # Get the top 10 highest positive changes for the day
    top_positive_changes = df_signal_strength_sma[df_signal_strength_mv > 0].nlargest(10)
    # If there are positive changes, mark them in the top trades DataFrame
    if not top_positive_changes.empty:
        df_top_trades_mv.loc[date, top_positive_changes.index] = 1

# Display the resulting DataFrame
print("Top Trades DataFrame MV (1 = Top 10 positive trades, 0 = Not a top trade):")
print(df_top_trades_mv)

# Calculate the returns based on buy signals
df_total_returns1 = (df_top_trades_mv/10) * df_price_diffs
# Sum the returns for each stock
total_return_per_stock1 = df_total_returns1.sum()
# Calculate the total return across all stocks
total_return_all_stocks1 = df_total_returns1.sum().sum()  # Sum all the values in the DataFrame
print("Total return for each stock MV:")
print(total_return_per_stock1)
print("\nTotal return for all stocks combined MV:")
print(total_return_all_stocks1)

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




# %%

print('---> Python Script End', t1 := datetime.datetime.now())
print('---> Total time taken', t1 - t0)


