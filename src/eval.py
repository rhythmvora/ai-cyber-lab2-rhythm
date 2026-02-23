"""
Model evaluation module
"""

import json
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import os
from src.data import get_processed_data


def load_model(filepath='results/model.joblib'):
    """
    Load trained model and scaler
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        tuple: (model, scaler)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found at {filepath}")
    
    artifacts = joblib.load(filepath)
    print(f"✓ Model loaded from: {filepath}")
    return artifacts['model'], artifacts['scaler']


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test set
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Evaluation metrics and predictions
    """
    print("\nEvaluating model on test set...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='binary')),
        'recall': float(recall_score(y_test, y_pred, average='binary')),
        'f1_score': float(f1_score(y_test, y_pred, average='binary'))
    }
    
    print("✓ Evaluation completed")
    return metrics, y_pred


def save_metrics(metrics, filepath='results/metrics.json'):
    """
    Save evaluation metrics to JSON file
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save metrics
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"✓ Metrics saved to: {filepath}")


def plot_confusion_matrix(y_test, y_pred, filepath='results/confusion_matrix.png'):
    """
    Plot and save confusion matrix
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        filepath: Path to save the plot
    """
    print("\nGenerating confusion matrix visualization...")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Legitimate', 'Phishing'],
        yticklabels=['Legitimate', 'Phishing'],
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Phishing Detection', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to: {filepath}")


def print_detailed_report(y_test, y_pred):
    """
    Print detailed classification report
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
    """
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(
        y_test, y_pred,
        target_names=['Legitimate', 'Phishing'],
        digits=4
    ))


def main():
    """
    Main evaluation pipeline
    """
    print("\n" + "="*70)
    print("EVALUATION PIPELINE")
    print("="*70)
    
    # Step 1: Load test data
    print("\n[STEP 1/5] Loading test data...")
    _, X_test, _, y_test, _ = get_processed_data()
    
    # Step 2: Load model
    print("\n[STEP 2/5] Loading trained model...")
    model, scaler = load_model()
    
    # Step 3: Evaluate model
    print("\n[STEP 3/5] Evaluating model...")
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    
    # Step 4: Display and save metrics
    print("\n[STEP 4/5] Results:")
    print("-"*70)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print("-"*70)
    
    save_metrics(metrics)
    
    # Step 5: Generate visualizations
    print("\n[STEP 5/5] Generating visualizations...")
    plot_confusion_matrix(y_test, y_pred)
    
    # Print detailed report
    print_detailed_report(y_test, y_pred)
    
    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  • results/metrics.json")
    print("  • results/confusion_matrix.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
