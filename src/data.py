"""
Data loading and preprocessing module - Updated for PhiUSIIL Dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


def load_dataset(filepath='data/raw/PhiUSIIL_Phishing_URL_Dataset.csv'):
    """
    Load the PhiUSIIL phishing dataset from CSV file

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame: Loaded dataset
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df):
    """
    Clean the PhiUSIIL dataset by handling missing values and duplicates

    Args:
        df: Input DataFrame

    Returns:
        DataFrame: Cleaned dataset
    """
    initial_rows = df.shape[0]

    # Remove duplicates
    df = df.drop_duplicates()
    print(f"Removed {initial_rows - df.shape[0]} duplicate rows")

    # Drop FILENAME column if it exists (as suggested in dataset documentation)
    if 'FILENAME' in df.columns:
        df = df.drop(columns=['FILENAME'])
        print("Dropped FILENAME column")

    # Handle missing values
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        print(f"\nMissing values found: {missing_before}")
        print("\nMissing values per column:")
        missing_cols = df.isnull().sum()
        print(missing_cols[missing_cols > 0])

        # Strategy: Drop columns with >50% missing values
        threshold = len(df) * 0.5
        cols_to_drop = df.columns[df.isnull().sum() > threshold].tolist()
        if cols_to_drop:
            print(f"\nDropping columns with >50% missing: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)

        # For remaining missing values, use median imputation for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        missing_after = df.isnull().sum().sum()
        print(f"Missing values after cleaning: {missing_after}")
    else:
        print("No missing values found")

    return df


def prepare_features(df, target_column=None):
    """
    Prepare features and target for modeling

    Args:
        df: Input DataFrame
        target_column: Name of the target column (auto-detected if None)

    Returns:
        tuple: (X, y, target_column_name) feature matrix, target vector, and target name
    """
    # Auto-detect target column if not specified
    if target_column is None:
        possible_targets = ['label', 'target', 'phishing', 'CLASS_LABEL', 'class', 'Label']
        for col in possible_targets:
            if col in df.columns:
                target_column = col
                print(f"Target column auto-detected: '{target_column}'")
                break

        if target_column is None:
            # Last column is often the target
            target_column = df.columns[-1]
            print(f"Using last column as target: '{target_column}'")

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Convert target to binary 0/1 if needed
    unique_values = y.unique()
    if len(unique_values) == 2:
        # Map to 0 and 1
        if set(unique_values) == {0, 1}:
            pass  # Already in correct format
        elif set(unique_values) == {-1, 1}:
            y = y.map({-1: 0, 1: 1})
            print("Converted target from {-1, 1} to {0, 1}")
        elif set(unique_values) == {'legitimate', 'phishing'}:
            y = y.map({'legitimate': 0, 'phishing': 1})
            print("Converted target from {'legitimate', 'phishing'} to {0, 1}")
        elif set(unique_values) == {'benign', 'malicious'}:
            y = y.map({'benign': 0, 'malicious': 1})
            print("Converted target from {'benign', 'malicious'} to {0, 1}")
        else:
            # Generic mapping: first unique value -> 0, second -> 1
            mapping = {unique_values[0]: 0, unique_values[1]: 1}
            y = y.map(mapping)
            print(f"Converted target from {set(unique_values)} to {{0, 1}}")

    # Ensure all features are numeric
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"\nWarning: Non-numeric columns found: {non_numeric}")
        print("These will be dropped as they cannot be used in the model")
        X = X.select_dtypes(include=[np.number])

    print(f"\nFeatures shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"\nTarget distribution:")
    print(f"Class 0 (Legitimate): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
    print(f"Class 1 (Phishing): {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")

    return X, y, target_column


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Split data into train/test sets and scale features

    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of test set
        random_state: Random seed

    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # Split the data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_processed_data(filepath='data/raw/PhiUSIIL_Phishing_URL_Dataset.csv', sample_size=None):
    """
    Complete data pipeline: load, clean, prepare, and split

    Args:
        filepath: Path to the raw dataset
        sample_size: If specified, randomly sample this many rows for faster training
                    (useful for large datasets like PhiUSIIL)

    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    print("="*70)
    print("DATA PROCESSING PIPELINE - PhiUSIIL Dataset")
    print("="*70)

    # Load data
    print("\n1. Loading data...")
    df = load_dataset(filepath)

    # Optional sampling for faster experimentation
    if sample_size and sample_size < len(df):
        print(f"\n   Sampling {sample_size} rows for faster processing...")
        df = df.sample(n=sample_size, random_state=42)
        print(f"   Dataset size after sampling: {df.shape[0]} rows")

    # Clean data
    print("\n2. Cleaning data...")
    df = clean_data(df)

    # Prepare features
    print("\n3. Preparing features...")
    X, y, target_col = prepare_features(df)

    # Split and scale
    print("\n4. Splitting and scaling...")
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    print("\n" + "="*70)
    print("DATA PROCESSING COMPLETED!")
    print("="*70)
    print(f"Final feature count: {X_train.shape[1]}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    # Test the data pipeline
    # Use sample_size for faster testing (remove for full dataset)
    X_train, X_test, y_train, y_test, scaler = get_processed_data(sample_size=10000)
    print("\n✓ Data pipeline test successful!")
    print(f"\nYou can now train the model with {X_train.shape[0]} training samples")
