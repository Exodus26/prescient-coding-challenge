import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Get the current working directory
current_directory = os.getcwd()
current_directory = os.path.join(current_directory, 'Cable')

# Construct the relative file path
# Assuming your CSV is located in a folder named 'data' within the current directory
relative_path = os.path.join(current_directory, 'returns_train.csv')

# Load the CSV file into a DataFrame
data = pd.read_csv(relative_path)

# Check the structure of the DataFrame
print(data.head())

print(data.columns)

data['Dates'] = pd.to_datetime(data['month_end'])

# Set the Date as the index
data.set_index('Dates', inplace=True)

# Plot the stock prices
plt.figure(figsize=(12, 6))
# plt.plot(data.index, data.loc[:,['Stock1','Stock10','Stock3']])
plt.plot(data.index, data.loc[:,['Stock1']])
plt.title('Stock Prices Over Time')
plt.xlabel('Dates')
plt.ylabel('Changes')
plt.legend()
plt.grid()

changes1 = data['Stock1']
print(np.mean(data['Stock1']))
print(len(changes1))

max_length = 20
min_length = 3
results = pd.DataFrame()

for r in range(min_length, max_length + 1):
    temp = []
    for i in range(0, len(changes1) - r + 1):  # Correct range to include valid indices
        values = np.sum(changes1[i:i+r])
        temp.append(values)
    
    # Add the new column to results DataFrame
    results[f"Version {r}"] = pd.Series(temp)  # Convert temp to Series to ensure proper length alignment

print(results)

for column in results:
    print(f"{column} sum: {np.mean(results[column])}")

# Identify points where the change is positive followed by negative or vice versa
# A positive change followed by a negative change
positive_to_negative = (changes1 > 0) & (changes1.shift(-1) < 0)

# A negative change followed by a positive change
negative_to_positive = (changes1 < 0) & (changes1.shift(-1) > 0)

# Combine both conditions
points_of_interest = changes1[positive_to_negative | negative_to_positive]

# Display the points where sign changes occur
print(points_of_interest)

plt.scatter(points_of_interest.index,points_of_interest, color = 'Red')
plt.show()