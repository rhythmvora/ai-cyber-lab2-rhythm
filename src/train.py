"""
Model training module
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
import os
from src.data import get_processed_data


def get_baseline_model(model_type='random_forest'):
    """
    Get a baseline model for training
    
    Args:
        model_type: Type of model ('random_forest', 'logistic', 'svm')
        
    Returns:
        Untrained model object
    """
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            n_jobs=-1
        )
        print(f"✓ Model: Random Forest (n_estimators=100, max_depth=10)")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def train_model(X_train, y_train, model_type='random_forest'):
    """
    Train a baseline model
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model to train
        
    Returns:
        Trained model
    """
    print(f"\n[Training {model_type.replace('_', ' ').title()}]")
    model = get_baseline_model(model_type)
    
    print("Training in progress... (this may take 2-5 minutes)")
    model.fit(X_train, y_train)
    print("✓ Training completed!")
    
    return model


def save_model(model, scaler, filepath='results/model.joblib'):
    """
    Save trained model and scaler
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        filepath: Path to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    artifacts = {
        'model': model,
        'scaler': scaler
    }
    joblib.dump(artifacts, filepath)
    print(f"✓ Model saved to: {filepath}")


def main():
    """
    Main training pipeline
    """
    print("\n" + "="*70)
    print("TRAINING PIPELINE")
    print("="*70)
    
    # Step 1: Load and prepare data
    print("\n[STEP 1/3] Loading and preparing data...")
    X_train, X_test, y_train, y_test, scaler = get_processed_data()
    
    # Step 2: Train model
    print("\n[STEP 2/3] Training model...")
    model = train_model(X_train, y_train, model_type='random_forest')
    
    # Step 3: Save model
    print("\n[STEP 3/3] Saving model...")
    save_model(model, scaler)
    
    # Quick accuracy check
    train_score = model.score(X_train, y_train)
    print(f"\n{'='*70}")
    print(f"Training Accuracy: {train_score:.4f} ({train_score*100:.2f}%)")
    print("="*70)
    
    print("\n✓ TRAINING COMPLETED SUCCESSFULLY!")
    print("→ Next step: Run evaluation with 'python -m src.eval'\n")


if __name__ == "__main__":
    main()
