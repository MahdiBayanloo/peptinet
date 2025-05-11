import os
import pandas as pd
import numpy as np

data_dir = "C:/Users/MahadiBayanloo/Documents/ZIB/download"
files    = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))

data_list = []
for fname in files:
    df = pd.read_csv(os.path.join(data_dir, fname))  # header inferred
    # if it's a single row of 1000 columns, df.values is shape (1,1000)
    vec = df.values.flatten()                        # -> shape (1000,)
    assert vec.shape == (1000,), f"{fname} has wrong shape {vec.shape}"
    data_list.append(vec)

X = np.stack(data_list, axis=0)                     # (256,1000)
print("X shape:", X.shape)
print(X)  # print the first row


# Normalize the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()        # zero mean, unit variance
X_scaled = scaler.fit_transform(X)
print("X shape:", X.shape)
print(X)  # print the first row

