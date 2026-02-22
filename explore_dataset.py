import pandas as pd

# Load the dataset
df = pd.read_csv('data/raw/PhiUSIIL_Phishing_URL_Dataset.csv')

print("Dataset Shape:", df.shape)
print("\n" + "="*70)
print("Column Names:")
print("="*70)
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print("\n" + "="*70)
print("First few rows:")
print("="*70)
print(df.head())

print("\n" + "="*70)
print("Data Types:")
print("="*70)
print(df.dtypes)

print("\n" + "="*70)
print("Target Variable Info:")
print("="*70)
# The target is usually named 'label', 'target', 'phishing', or similar
# Check for common target column names
possible_targets = ['label', 'target', 'phishing', 'CLASS_LABEL', 'class']
for col in possible_targets:
    if col in df.columns:
        print(f"\nTarget column found: '{col}'")
        print(df[col].value_counts())
        break

print("\n" + "="*70)
print("Missing Values:")
print("="*70)
print(df.isnull().sum().sum(), "total missing values")

print("\n" + "="*70)
print("Statistical Summary:")
print("="*70)
print(df.describe())
