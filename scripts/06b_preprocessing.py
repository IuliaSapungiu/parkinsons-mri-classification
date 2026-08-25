import pandas as pd
import os

# Define local path
DATA_DIR = r"D:\UoS\1DISSERTATION\PD\processed_data\extracted_features"
train_path = os.path.join(DATA_DIR, "train_rf_features.csv")

# Load dataset
df = pd.read_csv(train_path)

print("=== DATA PREPROCESSING INTEGRITY CHECK ===")

# 1. Check for Duplicate Subject IDs
duplicates = df['IDNUM'].duplicated().sum()
print(f"1. Duplicate rows based on IDNUM: {duplicates}")

# 2. Check for missing values across ALL columns
missing_total = df.isnull().sum().sum()
if missing_total > 0:
    print(f"2. [ALERT] Found {missing_total} total missing values in the dataset!")
    # Print exactly which columns have missing data
    missing_cols = df.isnull().sum()
    print(missing_cols[missing_cols > 0])
else:
    print("2. No missing values found in any column.")

# 3. Check for non-numeric (text) columns that need encoding
text_cols = df.select_dtypes(include=['object']).columns.tolist()
# Remove IDNUM or paths if they are strings
text_cols = [c for c in text_cols if c not in ['IDNUM']]
print(f"3. Text/Categorical columns that need encoding: {text_cols}")