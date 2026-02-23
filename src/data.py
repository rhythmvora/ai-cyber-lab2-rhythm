"""
Data loading and preprocessing module for PhiUSIIL dataset
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
    print(f"✓ Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def clean_data(df):
    """
    Clean the PhiUSIIL dataset
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame: Cleaned dataset
    """
    initial_rows = df.shape[0]
    
    # Remove duplicates
    df = df.drop_duplicates()
    removed = initial_rows - df.shape[0]
    if removed > 0:
        print(f"✓ Removed {removed:,} duplicate rows")
    else:
        print("✓ No duplicate rows found")
    
    # Drop FILENAME column if it exists
    if 'FILENAME' in df.columns:
        df = df.drop(columns=['FILENAME'])
        print("✓ Dropped FILENAME column")
    
    # Handle missing values
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        print(f"⚠ Found {missing_before:,} missing values")
        # Drop columns with >50% missing
        threshold = len(df) * 0.5
        cols_to_drop = df.columns[df.isnull().sum() > threshold].tolist()
        if cols_to_drop:
            print(f"✓ Dropping columns: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        
        # Fill remaining with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        print(f"✓ Filled missing values with median")
    else:
        print("✓ No missing values found")
    
    return df


def prepare_features(df, target_column=None):
    """
    Prepare features and target for modeling
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column (auto-detected if None)
        
    Returns:
        tuple: (X, y, target_column_name)
    """
    # Auto-detect target column
    if target_column is None:
        possible_targets = ['label', 'target', 'phishing', 'CLASS_LABEL', 'class', 'Label']
        for col in possible_targets:
            if col in df.columns:
                target_column = col
                print(f"✓ Target column detected: '{target_column}'")
                break
        
        if target_column is None:
            target_column = df.columns[-1]
            print(f"✓ Using last column as target: '{target_column}'")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Convert target to binary 0/1
    unique_values = sorted(y.unique())
    if len(unique_values) == 2:
        if set(unique_values) == {0, 1}:
            pass  # Already correct format
        elif set(unique_values) == {-1, 1}:
            y = y.map({-1: 0, 1: 1})
            print("✓ Converted target from {-1, 1} to {0, 1}")
        else:
            # Generic mapping
            y = y.map({unique_values[0]: 0, unique_values[1]: 1})
            print(f"✓ Converted target to binary: {unique_values[0]}→0, {unique_values[1]}→1")
    
    # Ensure all features are numeric
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"⚠ Dropping non-numeric columns: {non_numeric}")
        X = X.select_dtypes(include=[np.number])
    
    print(f"\n✓ Features shape: {X.shape}")
    print(f"✓ Number of features: {X.shape[1]}")
    print(f"\n✓ Target distribution:")
    print(f"  Class 0 (Legitimate): {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.1f}%)")
    print(f"  Class 1 (Phishing):   {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.1f}%)")
    
    return X, y, target_column


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Split data into train/test sets and scale features
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n✓ Data split completed:")
    print(f"  Training set: {X_train.shape[0]:,} samples")
    print(f"  Test set:     {X_test.shape[0]:,} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Feature scaling applied (StandardScaler)")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_processed_data(filepath='data/raw/PhiUSIIL_Phishing_URL_Dataset.csv', sample_size=None):
    """
    Complete data pipeline: load, clean, prepare, and split
    
    Args:
        filepath: Path to the raw dataset
        sample_size: Optional - sample this many rows for faster training
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    print("\n" + "="*70)
    print("DATA PROCESSING PIPELINE - PhiUSIIL Dataset")
    print("="*70 + "\n")
    
    # Step 1: Load data
    print("[1/4] Loading dataset...")
    df = load_dataset(filepath)
    
    # Optional sampling
    if sample_size and sample_size < len(df):
        print(f"\n⚠ Sampling {sample_size:,} rows for faster processing...")
        df = df.sample(n=sample_size, random_state=42)
        print(f"✓ Sampled dataset: {df.shape[0]:,} rows")
    
    # Step 2: Clean data
    print(f"\n[2/4] Cleaning dataset...")
    df = clean_data(df)
    
    # Step 3: Prepare features
    print(f"\n[3/4] Preparing features...")
    X, y, target_col = prepare_features(df)
    
    # Step 4: Split and scale
    print(f"\n[4/4] Splitting and scaling...")
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    
    print("\n" + "="*70)
    print("✓ DATA PROCESSING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nDataset Summary:")
    print(f"  • Total samples: {len(df):,}")
    print(f"  • Features: {X.shape[1]}")
    print(f"  • Training samples: {len(y_train):,}")
    print(f"  • Test samples: {len(y_test):,}")
    print("="*70 + "\n")
    
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    # Test the data pipeline
    print("Testing data pipeline...")
    X_train, X_test, y_train, y_test, scaler = get_processed_data()
    print("\n✓ Data pipeline test successful!")