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
print(df_pivoted_prices)

df_price_diffs = df_pivoted_prices.diff()
print(df_price_diffs)





# %%
