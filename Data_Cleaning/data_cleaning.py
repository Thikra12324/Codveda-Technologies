import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("3) Sentiment dataset.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Remove unwanted columns (if they exist)
columns_to_remove = ['Unnamed: 0', 'Unnamed: 0.1']

for col in columns_to_remove:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Fill missing numerical values with mean
num_cols = df.select_dtypes(include='number').columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

# Fill missing categorical values with mode
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Convert Timestamp to datetime
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Remove extra spaces from text columns
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip()

# Convert Sentiment values to uppercase
if 'Sentiment' in df.columns:
    df['Sentiment'] = df['Sentiment'].str.upper()

# Check missing values again
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv("cleaned_sentiment_dataset.csv", index=False)

print("\nData Cleaning Completed Successfully!")
print("Cleaned file saved as: cleaned_sentiment_dataset.csv")
