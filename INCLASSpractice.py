'''
"NUMPY IN CLASS PRACTICE"
import numpy as np

ages = np.array([39, 50, 38, 53, 28, 37, 49, 52, 31, 42])
# TODO: Calculate mean age
mean_age = np.mean(ages)

# TODO: Find ages over 40
over_40 = ages[ages > 40]

# TODO: Count how many are over 40
count_over_40 = np.sum(ages > 40)

print(f"Mean age: {mean_age:.1f}")
print(f"Ages over 40: {over_40}")
print(f"Count over 40: {count_over_40}")
'''
'''
# Task 1: Create binary encoding for income (target variable)
# >50K should be 1, <=50K should be 0

df_clean['income_binary'] = (df_clean['income'] == '>50K').astype(int)
print("Income encoding:")
print(df_clean[['income', 'income_binary']].drop_duplicates())

# Task 2: How many high earners (income_binary = 1) are there?
print(f"\nHigh earners: {df_clean['income_binary'].sum()}")
print(f"Percentage: {df_clean['income_binary'].mean() * 100:.1f}%")
'''

import matplotlib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import pandas as pd

# Create a small dataset with outliers
data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 90],  # 90 is an outlier
    'income': [30000, 45000, 55000, 65000, 80000, 200000]  # 200000 is an outlier
})

print("Original data:")
print(data)

# Task 1: Apply MinMaxScaler
minmax = MinMaxScaler()
data_minmax = pd.DataFrame(minmax.fit_transform(data), columns=data.columns)
print("\nMinMaxScaler result:")
print(data_minmax.round(3))

# Task 2: Apply RobustScaler
robust = RobustScaler()
data_robust = pd.DataFrame(robust.fit_transform(data), columns=data.columns)
print("\nRobustScaler result:")
print(data_robust.round(3))

# Question: Which scaler handled the outliers better?
# The RobustScaler handled the outliers better as it is less affected by extreme values.   


import pandas as pd
import numpy as npimport 
matplotlib.pyplot as plt
#**Questions to answer:**
#1. Is the distribution symmetric or skewed? 
#2. What's the typical work week?
#3. Are there outliers?