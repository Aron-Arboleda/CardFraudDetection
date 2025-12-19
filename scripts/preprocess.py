"""
Credit Card Fraud Detection - Data Preprocessing Script
Loads raw data, splits train/test, standardizes features, applies SMOTE, and saves processed data.

This script performs:
1. Load creditcard.csv
2. Split 80/20 stratified (preserves class imbalance ratio)
3. Standardize features using StandardScaler (fit on train only)
4. Apply SMOTE to training set only (balances classes)
5. Pickle processed train/test sets for model training

Usage:
    python scripts/preprocess.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_data():
    """Load the credit card fraud dataset."""
    data_path = "data/creditcard.csv"
    
    print("Step 1: Loading dataset...")
    if not os.path.exists(data_path):
        print(f"✗ Error: Dataset not found at {data_path}")
        print("Please run 'python scripts/download_data.py' first.")
        sys.exit(1)
    
    try:
        df = pd.read_csv(data_path)
        print(f"✓ Dataset loaded successfully")
        print(f"  - Shape: {df.shape}")
        print(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        return df
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        sys.exit(1)

def explore_data(df):
    """Display basic information about the dataset."""
    print("\nStep 2: Exploring dataset...")
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    print(f"✓ Missing values: {missing}")
    
    # Class distribution
    class_dist = df['Class'].value_counts()
    fraud_pct = (class_dist[1] / len(df)) * 100
    
    print(f"✓ Class distribution:")
    print(f"  - Normal (0): {class_dist[0]:,} ({100-fraud_pct:.3f}%)")
    print(f"  - Fraud (1):  {class_dist[1]:,} ({fraud_pct:.3f}%)")
    print(f"  - Imbalance ratio: 1:{class_dist[0]/class_dist[1]:.0f}")

def split_data(df, test_size=0.2):
    """Split data into train and test sets with stratification."""
    print(f"\nStep 3: Splitting data (train: {int((1-test_size)*100)}%, test: {int(test_size*100)}%)...")
    
    # # --- ADD THIS PART HERE ---
    # from sklearn.utils import shuffle
    # df = shuffle(df, random_state=RANDOM_SEED)
    # # --------------------------
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Stratified split to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y
    )
    
    print(f"✓ Data split completed")
    print(f"  - Training set: {X_train.shape[0]:,} samples")
    print(f"  - Test set: {X_test.shape[0]:,} samples")
    print(f"  - Train fraud cases: {y_train.sum():,} ({(y_train.sum()/len(y_train)*100):.3f}%)")
    print(f"  - Test fraud cases: {y_test.sum():,} ({(y_test.sum()/len(y_test)*100):.3f}%)")
    
    return X_train, X_test, y_train, y_test

def standardize_features(X_train, X_test):
    """Standardize features using StandardScaler (fit on train only)."""
    print("\nStep 4: Standardizing features...")
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit on training data only
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index
    )
    
    print("✓ Features standardized (fit on train, applied to both)")
    print(f"  - Mean of training features: ~{X_train_scaled.mean().mean():.6f}")
    print(f"  - Std of training features: ~{X_train_scaled.std().mean():.6f}")
    
    return X_train_scaled, X_test_scaled, scaler

def apply_smote(X_train, y_train):
    """Apply SMOTE to balance the training set."""
    print("\nStep 5: Applying SMOTE to training set...")
    print(f"  - Before SMOTE: {Counter(y_train)}")
    
    # Initialize SMOTE
    smote = SMOTE(random_state=RANDOM_SEED, n_jobs=-1)
    
    try:
        # Apply SMOTE
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"  - After SMOTE: {Counter(y_train_resampled)}")
        # Create visualization of class balance after SMOTE
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        classes = ['Normal (0)', 'Fraud (1)']
        counts = [Counter(y_train_resampled)[0], Counter(y_train_resampled)[1]]
        colors = ['#2ecc71', '#e74c3c']

        ax.bar(classes, counts, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title('Class Distribution After SMOTE', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(counts) * 1.1)

        for i, count in enumerate(counts):
            ax.text(i, count + max(counts)*0.02, f'{count:,}\n(50%)', ha='center', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig('data/smote_balance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Visualization saved: data/smote_balance.png")
        print(f"✓ SMOTE applied successfully")
        print(f"  - New training set size: {X_train_resampled.shape[0]:,} samples")
        print(f"  - Balanced classes: 50% fraud, 50% normal")
        
        return X_train_resampled, y_train_resampled
    except Exception as e:
        print(f"✗ Error applying SMOTE: {e}")
        sys.exit(1)

def create_validation_split(X_train, y_train, val_size=0.2):
    """Create stratified validation split from training data."""
    print(f"\nStep 6: Creating stratified validation split ({int((1-val_size)*100)}% train, {int(val_size*100)}% validation)...")
    
    # Stratified split to preserve class distribution in both train and validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        random_state=RANDOM_SEED,
        stratify=y_train
    )
    
    print(f"✓ Validation split completed")
    print(f"  - Training samples: {X_train_split.shape[0]:,}")
    print(f"  - Validation samples: {X_val.shape[0]:,}")
    print(f"  - Train fraud cases: {y_train_split.sum():,} ({(y_train_split.sum()/len(y_train_split)*100):.3f}%)")
    print(f"  - Validation fraud cases: {y_val.sum():,} ({(y_val.sum()/len(y_val)*100):.3f}%)")
    
    return X_train_split, X_val, y_train_split, y_val

def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, scaler):
    """Save processed data to pickle files."""
    print("\nStep 7: Saving processed data...")
    
    # Create data directory if it doesn't exist
    Path("data").mkdir(parents=True, exist_ok=True)
    
    # Prepare data dictionaries
    train_data = {
        'X_train': X_train,
        'y_train': y_train
    }
    
    val_data = {
        'X_val': X_val,
        'y_val': y_val
    }
    
    test_data = {
        'X_test': X_test,
        'y_test': y_test
    }
    
    scaler_data = {
        'scaler': scaler
    }
    
    # Save to pickle files
    try:
        with open('data/train_data.pkl', 'wb') as f:
            pickle.dump(train_data, f)
        print("✓ Saved: data/train_data.pkl")
        
        with open('data/val_data.pkl', 'wb') as f:
            pickle.dump(val_data, f)
        print("✓ Saved: data/val_data.pkl")
        
        with open('data/test_data.pkl', 'wb') as f:
            pickle.dump(test_data, f)
        print("✓ Saved: data/test_data.pkl")
        
        with open('data/scaler.pkl', 'wb') as f:
            pickle.dump(scaler_data, f)
        print("✓ Saved: data/scaler.pkl")
        
        # Calculate and display file sizes
        train_size = os.path.getsize('data/train_data.pkl') / (1024**2)
        val_size = os.path.getsize('data/val_data.pkl') / (1024**2)
        test_size = os.path.getsize('data/test_data.pkl') / (1024**2)
        scaler_size = os.path.getsize('data/scaler.pkl') / (1024**2)
        
        print(f"\nFile sizes:")
        print(f"  - train_data.pkl: {train_size:.2f} MB")
        print(f"  - val_data.pkl: {val_size:.2f} MB")
        print(f"  - test_data.pkl: {test_size:.2f} MB")
        print(f"  - scaler.pkl: {scaler_size:.2f} MB")
        
        return True
    except Exception as e:
        print(f"✗ Error saving data: {e}")
        return False

def main():
    """Main execution function."""
    print("=" * 70)
    print("Credit Card Fraud Detection - Data Preprocessing")
    print("=" * 70)
    print()
    
    # Load data
    df = load_data()
    
    # Explore data
    explore_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
    
    # Standardize features
    X_train_scaled, X_test_scaled, scaler = standardize_features(X_train, X_test)
    
    # Apply SMOTE to training set only
    X_train_resampled, y_train_resampled = apply_smote(X_train_scaled, y_train)
    
    # Create stratified validation split
    X_train_final, X_val, y_train_final, y_val = create_validation_split(
        X_train_resampled, y_train_resampled, val_size=0.2
    )
    
    # Save processed data
    if save_processed_data(X_train_final, X_val, X_test_scaled, 
                          y_train_final, y_val, y_test, scaler):
        print()
        print("=" * 70)
        print("✅ SUCCESS! Data preprocessing completed")
        print("=" * 70)
        print("\nProcessed data summary:")
        print(f"  - Training samples: {len(X_train_final):,} (balanced with SMOTE)")
        print(f"  - Validation samples: {len(X_val):,} (balanced with SMOTE, stratified)")
        print(f"  - Test samples: {len(X_test_scaled):,} (original imbalanced distribution)")
        print(f"  - Features: {X_train_final.shape[1]}")
        print("\nNext steps:")
        print("1. Run data exploration: jupyter notebook notebooks/data_exploration.ipynb")
        print("2. Train model locally or upload to Google Colab:")
        print("   - Local: jupyter notebook notebooks/model_training.ipynb")
        print("   - Colab: Upload notebooks/model_training.ipynb to Google Colab")
        print("\nFiles ready for training:")
        print("  ✓ data/train_data.pkl")
        print("  ✓ data/val_data.pkl")
        print("  ✓ data/test_data.pkl")
        print("  ✓ data/scaler.pkl")
    else:
        print()
        print("=" * 70)
        print("❌ FAILED to complete preprocessing")
        print("=" * 70)
        sys.exit(1)

if __name__ == "__main__":
    main()