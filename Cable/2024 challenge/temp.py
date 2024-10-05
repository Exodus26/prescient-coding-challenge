import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Lambda


# Function to identify optimal buy/sell points based on cumulative changes for maximum profit
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
    top_trades_df = top_trades_df.shift(-1)  # Shift rows up by 1
    top_trades_df = top_trades_df.dropna()  # Shift rows up by 1

    # Prepare the data for LSTM
    df_training_answer = top_trades_df.loc[:'2023-12-31']
    df_training_input = df_price_diffs.loc[:'2023-12-31']

    df_testing_input = top_trades_df.loc['2024-01-01':'2024-06-20']
    df_testing_answer = df_price_diffs.loc['2024-01-01':'2024-06-20']

     # Replace NaN values with 0 in all DataFrames
    df_training_input.fillna(0, inplace=True)
    df_training_answer.fillna(0, inplace=True)
    df_testing_input.fillna(0, inplace=True)
    df_testing_answer.fillna(0, inplace=True)

    
    # Define parameters
    time_step = 15  # Number of previous days to use for predicting the next day

    # Create datasets
    def create_dataset(data, labels, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), :])  # Add the sequence of input features
            y.append(labels[i + time_step, :])    # Add the next value as the label (target)
        return np.array(X), np.array(y)

    # Prepare the training and testing datasets
    X_train, Y_train = create_dataset(df_training_input.values, df_training_answer.values, time_step)
    X_test, Y_test = create_dataset(df_testing_input.values, df_testing_answer.values, time_step)

    # Reshape input to be [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])


    # Function to apply top-k selection post-processing
    def apply_top_k_selection(predictions, k=10):
        # Get the top k indices
        top_k_indices = tf.argsort(predictions, axis=-1, direction='DESCENDING')[:, :k]
        
        # Create a mask filled with zeros of shape [batch_size, num_stocks]
        batch_size = tf.shape(predictions)[0]
        num_stocks = tf.shape(predictions)[1]
        mask = tf.zeros([batch_size, num_stocks], dtype=tf.float32)
        
        # Create a tensor of ones for the top k
        updates = tf.ones([batch_size, k], dtype=tf.float32)
        
        # For each batch, scatter the ones at the top k indices
        for i in range(batch_size):
            mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(top_k_indices[i], axis=-1), updates[i])

        return mask

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(Y_train.shape[1]))  # Number of stocks/columns

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.2, 
                        callbacks=[early_stopping], verbose=1)

    # Evaluate the model
    train_loss = model.evaluate(X_train, Y_train)
    test_loss = model.evaluate(X_test, Y_test)

    print(f'Train Loss: {train_loss}')
    print(f'Test Loss: {test_loss}')

    # Make predictions
    predictions = model.predict(X_test)

    def array_k(predictions, k=10):
        # Initialize the output mask with zeros
        mask = np.zeros_like(predictions)
        # Iterate over each row in the predictions
        for i in range(predictions.shape[0]):
            # Get the indices of the top k values
            top_k_indices = np.argsort(predictions[i])[-k:]  # Get indices of the k largest elements
            mask[i, top_k_indices] = 1  # Set those indices to 1 in the mask

        return mask
    
    # Apply post-processing to select top 10
    final_outputs = array_k(predictions)

    # Now final_outputs contains the processed predictions with exactly 10 ones per sample

    # Plot the predictions without scaling
    plt.figure(figsize=(14, 5))
    for i in range(min(5, predictions.shape[1])):  # Plot only first 5 stocks
        plt.subplot(1, 5, i + 1)
        plt.plot(Y_test[:, i], label='True Value')
        plt.plot(predictions[:, i], label='Predicted Value')
        plt.title(f'Stock {i + 1}')
        plt.xlabel('Days')
        plt.ylabel('Indicator')
        plt.legend()
    plt.tight_layout()
    plt.show()

main()