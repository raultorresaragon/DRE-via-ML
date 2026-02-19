# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: real_data_analysis_00.py
# Date: 2025-01-08
# Note: This script imports and inspects
#       a data set provided by Dr. Ahn
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Clear environment
import sys
import os

def recode_NA(df, col):
    """Replace -1 with NaN in specified column"""
    df[col] = df[col].replace(-1, np.nan)
    return df

def recode_01(df, col, value1, value0=0):
    """Recode binary variables to 0/1"""
    df[col] = df[col].replace({value0: 0, value1: 1})
    return df

# Read data
df = pd.read_excel("real_data/ASTR_dataset.xlsx")
df.columns = ["id", "gender", "age", "time", "gh", "chemo", "partial_or_total_removal"]

# Basic info
print(f"N: {df.shape[0]}")
print(f"Number of participants: {df['id'].nunique()}")

# Number of measurements per participant
measurements_per_id = df['id'].value_counts()
print("Measurements per participant:")
print(measurements_per_id.value_counts().sort_index())

# Recode -1 as missing
for col in df.columns:
    df = recode_NA(df, col)

# Recode binary variables
print("Gender counts:")
print(df['gender'].value_counts())
df = recode_01(df, "gender", value1="F", value0="M")

print("Chemo counts:")
print(df['chemo'].value_counts())
df = recode_01(df, "chemo", value1=2, value0=1)

print("Partial or total removal counts:")
print(df['partial_or_total_removal'].value_counts())
df = recode_01(df, "partial_or_total_removal", value1=2, value0=1)

# Keep only the first measurement (cross-sectional analysis)
df_t1 = df[df['time'] == 1].copy()

# Tabulate variables
print("\nFirst measurement data:")
print("Gender counts:")
print(df_t1['gender'].value_counts())

print("Chemo counts:")
print(df_t1['chemo'].value_counts())

print("Partial or total removal counts:")
print(df_t1['partial_or_total_removal'].value_counts())

# Summarize continuous variables
print("GH summary:")
print(df_t1['gh'].describe())

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(df_t1['gh'].dropna(), bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('GH')
plt.ylabel('Frequency')
plt.title('Distribution of GH scores')
plt.show()

# Output files
os.makedirs("real_data", exist_ok=True)
df.to_csv("real_data/recoded_ASTR.csv", index=False)
df_t1.to_csv("real_data/recoded_ASTR_t1.csv", index=False)

print("\nData preprocessing complete. Files saved:")
print("- real_data/recoded_ASTR.csv")
print("- real_data/recoded_ASTR_t1.csv")