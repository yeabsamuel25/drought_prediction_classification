"""
Data Preprocessing Module
Loads and prepares the drought dataset
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib


def load_data(filepath):
    """Load the drought dataset"""
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"✅ Data loaded: {len(df):,} rows, {len(df.columns)} columns")
    
    return df


def prepare_features_target(df):
    """Separate features (X) and target (y)"""
    print("\n" + "=" * 80)
    print("PREPARING FEATURES AND TARGET")
    print("=" * 80)
    
    # Drop row_id (not useful for prediction)
    if 'row_id' in df.columns:
        df = df.drop('row_id', axis=1)
    
    # Separate X (features) and y (target)
    X = df.drop('label', axis=1)
    y = df['label']
    
    print(f"✅ Features (X): {X.shape}")
    print(f"✅ Target (y): {y.shape}")
    print(f"   - No Drought (0): {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.1f}%)")
    print(f"   - Drought (1): {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.1f}%)")
    
    return X, y


def split_data(X, y):
    """Split into train/validation/test sets"""
    print("\n" + "=" * 80)
    print("SPLITTING DATA")
    print("=" * 80)
    
    # First split: separate test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Second split: separate validation from training (10% of original)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
    )
    
    print(f"✅ Training set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"✅ Validation set: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"✅ Test set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """Scale features to mean=0, std=1"""
    print("\n" + "=" * 80)
    print("SCALING FEATURES")
    print("=" * 80)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    
    print("✅ Features scaled (mean=0, std=1)")
    
    # Save the scaler for future predictions
    joblib.dump(scaler, 'models/scaler.pkl')
    print("✅ Scaler saved to: models/scaler.pkl")
    
    return X_train_scaled, X_val_scaled, X_test_scaled


def preprocess_pipeline(filepath):
    """Complete preprocessing pipeline"""
    print("\n" + "🚀" * 40)
    print("STARTING PREPROCESSING")
    print("🚀" * 40)
    
    # Step 1: Load
    df = load_data(filepath)
    
    # Step 2: Prepare
    X, y = prepare_features_target(df)
    
    # Step 3: Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Step 4: Scale
    X_train_scaled, X_val_scaled, X_test_scaled = scale_features(
        X_train, X_val, X_test
    )
    
    print("\n" + "✅" * 40)
    print("PREPROCESSING COMPLETE!")
    print("✅" * 40)
    
    return {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': X_train.columns.tolist()
    }