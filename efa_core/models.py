"""
Machine Learning Models Module
==============================

Functions for training and evaluating Random Forest classifiers.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def train_rf_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = None,
    **kwargs
) -> dict:
    """
    Train Random Forest classifier with cross-validation.

    Parameters:
        X: Feature DataFrame
        y: Target Series
        cv: Number of cross-validation folds. Defaults to config.RF_CV_FOLDS
        **kwargs: Override RF_PARAMS from config

    Returns:
        Dictionary with model, predictions, encoder, scaler, and metrics
    """
    if cv is None:
        cv = config.RF_CV_FOLDS

    # Merge default params with any overrides
    rf_params = {**config.RF_PARAMS, **kwargs}

    # Handle missing values
    X_clean = X.fillna(0)

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Create classifier
    rf = RandomForestClassifier(**rf_params)

    # Cross-validation predictions
    y_pred = cross_val_predict(rf, X_scaled, y_encoded, cv=cv)
    y_pred_labels = le.inverse_transform(y_pred)

    # Calculate cross-validation scores
    cv_scores = cross_val_score(rf, X_scaled, y_encoded, cv=cv)

    # Fit on all data for feature importance
    rf.fit(X_scaled, y_encoded)

    # Calculate accuracy
    accuracy = accuracy_score(y_encoded, y_pred)

    print("\n" + "=" * 60)
    print(f"RANDOM FOREST CLASSIFIER (cv={cv})")
    print("=" * 60)
    print(f"\nOverall accuracy: {accuracy*100:.1f}%")
    print(f"CV scores: {cv_scores.round(3)} (mean: {cv_scores.mean():.3f})")

    return {
        'model': rf,
        'predictions': y_pred_labels,
        'predictions_encoded': y_pred,
        'encoder': le,
        'scaler': scaler,
        'accuracy': accuracy,
        'cv_scores': cv_scores,
        'feature_names': list(X.columns),
    }


def get_feature_importance(
    model: RandomForestClassifier,
    feature_names: list[str]
) -> pd.DataFrame:
    """
    Extract feature importance rankings.

    Parameters:
        model: Fitted RandomForestClassifier
        feature_names: List of feature names

    Returns:
        DataFrame with features sorted by importance
    """
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nFeature Importance:")
    print("-" * 40)
    for _, row in importance.iterrows():
        bar = '█' * int(row['importance'] * 50)
        print(f"  {row['feature']}: {row['importance']:.3f} {bar}")

    return importance


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list = None
) -> dict:
    """
    Calculate classification metrics and confusion matrix.

    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        labels: Optional label names for confusion matrix

    Returns:
        Dictionary with metrics and confusion matrix
    """
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)

    if labels is not None:
        conf_matrix_df = pd.DataFrame(
            conf_matrix,
            index=[f'True_{l}' for l in labels],
            columns=[f'Pred_{l}' for l in labels]
        )
    else:
        conf_matrix_df = pd.DataFrame(conf_matrix)

    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'confusion_matrix_df': conf_matrix_df,
    }


def analyze_predictions(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str
) -> dict:
    """
    Analyze prediction accuracy at row level.

    Parameters:
        df: DataFrame with actual and predicted columns
        actual_col: Column with actual labels
        predicted_col: Column with predicted labels

    Returns:
        Dictionary with accuracy metrics by group
    """
    df = df.copy()
    df['MATCH'] = df[actual_col] == df[predicted_col]

    overall_accuracy = df['MATCH'].mean()

    # Per-group accuracy
    per_group = df.groupby(actual_col).agg({
        'MATCH': ['mean', 'sum', 'count']
    })
    per_group.columns = ['accuracy', 'correct', 'total']
    per_group = per_group.sort_values('total', ascending=False)

    print("\nPer-group accuracy:")
    for group in per_group.index:
        row = per_group.loc[group]
        print(f"  {group}: {row['accuracy']*100:.0f}% ({int(row['correct'])}/{int(row['total'])})")

    return {
        'overall_accuracy': overall_accuracy,
        'per_group': per_group,
    }


def identify_mismatches(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    id_col: str = None
) -> pd.DataFrame:
    """
    Identify rows where predictions don't match actuals.

    Parameters:
        df: DataFrame with actual and predicted columns
        actual_col: Column with actual labels
        predicted_col: Column with predicted labels
        id_col: Optional ID column for grouping

    Returns:
        DataFrame with mismatched records
    """
    df = df.copy()
    df['MATCH'] = df[actual_col] == df[predicted_col]

    mismatches = df[~df['MATCH']].copy()

    print(f"\nTotal mismatches: {len(mismatches)} ({len(mismatches)/len(df)*100:.1f}%)")

    if id_col and id_col in df.columns:
        # Aggregate by ID
        mismatch_by_id = mismatches.groupby(id_col).size().sort_values(ascending=False)
        print(f"\nTop entities with mismatches:")
        for entity, count in mismatch_by_id.head(10).items():
            total = len(df[df[id_col] == entity])
            print(f"  {entity}: {count}/{total} mismatches")

    return mismatches


def predict_with_model(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    encoder: LabelEncoder,
    X_new: pd.DataFrame
) -> np.ndarray:
    """
    Make predictions with fitted model.

    Parameters:
        model: Fitted RandomForestClassifier
        scaler: Fitted StandardScaler
        encoder: Fitted LabelEncoder
        X_new: New features to predict

    Returns:
        Array of predicted labels
    """
    X_clean = X_new.fillna(0)
    X_scaled = scaler.transform(X_clean)
    y_pred_encoded = model.predict(X_scaled)
    y_pred_labels = encoder.inverse_transform(y_pred_encoded)
    return y_pred_labels
