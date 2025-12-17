"""
Model Evaluation Module
Evaluates and compares all models with extended metrics
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, log_loss, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a single model with all metrics"""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities for ROC-AUC and Log Loss
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
        y_proba_positive = y_proba[:, 1]
        
        # Calculate Log Loss
        logloss = log_loss(y_test, y_proba)
    else:
        y_proba_positive = y_pred
        logloss = None
    
    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate all metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'Specificity': tn / (tn + fp),  # NEW!
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba_positive),
        'Log Loss': logloss,  # NEW!
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,  # NEW! Negative Predictive Value
        'Confusion Matrix': cm,
        'Classification Report': classification_report(
            y_test, y_pred, 
            target_names=['No Drought', 'Drought'],
            output_dict=True
        )
    }
    
    return metrics


def evaluate_all_models(models, X_test, y_test):
    """Evaluate all models"""
    print("\n" + "=" * 80)
    print("EVALUATING MODELS")
    print("=" * 80)
    
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        results[name] = evaluate_model(model, X_test, y_test, name)
    
    print("\n✅ All models evaluated!")
    
    return results


def print_results(results):
    """Print basic comparison table"""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    
    # Create comparison table
    data = []
    for name, metrics in results.items():
        data.append({
            'Model': name,
            'Accuracy': f"{metrics['Accuracy']:.4f}",
            'Precision': f"{metrics['Precision']:.4f}",
            'Recall': f"{metrics['Recall']:.4f}",
            'F1-Score': f"{metrics['F1-Score']:.4f}",
            'ROC-AUC': f"{metrics['ROC-AUC']:.4f}"
        })
    
    df = pd.DataFrame(data)
    print("\n" + df.to_string(index=False))
    
    # Find best model
    best = max(results.items(), key=lambda x: x[1]['F1-Score'])
    
    print("\n" + "=" * 80)
    print(f"🏆 BEST MODEL: {best[0]}")
    print("=" * 80)
    print(f"   Accuracy:  {best[1]['Accuracy']:.4f} ({best[1]['Accuracy']*100:.2f}%)")
    print(f"   Precision: {best[1]['Precision']:.4f}")
    print(f"   Recall:    {best[1]['Recall']:.4f}")
    print(f"   F1-Score:  {best[1]['F1-Score']:.4f}")
    print(f"   ROC-AUC:   {best[1]['ROC-AUC']:.4f}")
    
    return df


def print_extended_results(results):
    """Print extended evaluation metrics for all models"""
    print("\n" + "=" * 80)
    print("EXTENDED EVALUATION METRICS")
    print("=" * 80)
    
    for model_name, metrics in results.items():
        print(f"\n{'='*80}")
        print(f"📊 {model_name.upper()}")
        print(f"{'='*80}")
        
        # Basic Classification Metrics
        print(f"\n📈 BASIC METRICS:")
        print(f"   Accuracy:     {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)")
        print(f"   Precision:    {metrics['Precision']:.4f}")
        print(f"   Recall:       {metrics['Recall']:.4f}")
        print(f"   Specificity:  {metrics['Specificity']:.4f}")
        print(f"   F1-Score:     {metrics['F1-Score']:.4f}")
        print(f"   ROC-AUC:      {metrics['ROC-AUC']:.4f}")
        
        # Probabilistic Metric (if available)
        if metrics['Log Loss'] is not None:
            print(f"\n🎲 PROBABILISTIC METRIC:")
            print(f"   Log Loss:     {metrics['Log Loss']:.4f} (lower is better)")
        
        # Confusion Matrix Details
        cm = metrics['Confusion Matrix']
        tn, fp, fn, tp = cm.ravel()
        print(f"\n📊 CONFUSION MATRIX BREAKDOWN:")
        print(f"   True Negatives:  {tn:,} (Correctly predicted No Drought)")
        print(f"   False Positives: {fp:,} (Incorrectly predicted Drought)")
        print(f"   False Negatives: {fn:,} (Missed Droughts)")
        print(f"   True Positives:  {tp:,} (Correctly predicted Drought)")
        
        # Additional Derived Metrics
        print(f"\n🔍 ADDITIONAL METRICS:")
        print(f"   NPV (Negative Predictive Value): {metrics['NPV']:.4f}")
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        print(f"   False Positive Rate: {false_positive_rate:.4f}")
        print(f"   False Negative Rate: {false_negative_rate:.4f}")
        
        # Per-Class Performance
        clf_report = metrics['Classification Report']
        print(f"\n📋 PER-CLASS PERFORMANCE:")
        print(f"\n   No Drought Class:")
        print(f"      Precision: {clf_report['No Drought']['precision']:.4f}")
        print(f"      Recall:    {clf_report['No Drought']['recall']:.4f}")
        print(f"      F1-Score:  {clf_report['No Drought']['f1-score']:.4f}")
        print(f"      Support:   {clf_report['No Drought']['support']:,} samples")
        
        print(f"\n   Drought Class:")
        print(f"      Precision: {clf_report['Drought']['precision']:.4f}")
        print(f"      Recall:    {clf_report['Drought']['recall']:.4f}")
        print(f"      F1-Score:  {clf_report['Drought']['f1-score']:.4f}")
        print(f"      Support:   {clf_report['Drought']['support']:,} samples")


def plot_confusion_matrices(results):
    """Plot confusion matrices for all models"""
    print("\n" + "=" * 80)
    print("CREATING CONFUSION MATRICES")
    print("=" * 80)
    
    n_models = len(results)
    
    # For 4 models, use 2x2 grid
    if n_models == 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    else:
        # For other numbers, calculate dynamically
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    
    axes = axes.ravel()
    
    for idx, (name, metrics) in enumerate(results.items()):
        cm = metrics['Confusion Matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['No Drought', 'Drought'],
                    yticklabels=['No Drought', 'Drought'])
        
        axes[idx].set_title(f'{name}\nAccuracy: {metrics["Accuracy"]:.4f}',
                           fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    # Hide extra subplots if any
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Confusion Matrices - All Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
    
    print("✅ Saved to: results/confusion_matrices.png")


def plot_metrics_comparison(results):
    """Plot metrics comparison bar chart"""
    print("\n" + "=" * 80)
    print("CREATING METRICS COMPARISON")
    print("=" * 80)
    
    models = list(results.keys())
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    data = {metric: [] for metric in metrics_names}
    
    for model_name in models:
        for metric in metrics_names:
            data[metric].append(results[model_name][metric])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(models))
    width = 0.15
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    for idx, (metric, values) in enumerate(data.items()):
        offset = width * idx
        bars = ax.bar(x + offset, values, width, label=metric, 
                      color=colors[idx], alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Models', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/metrics_comparison.png', dpi=300, bbox_inches='tight')
    
    print("✅ Saved to: results/metrics_comparison.png")


def plot_extended_metrics(results):
    """Plot extended metrics comparison including Log Loss and Specificity"""
    print("\n" + "=" * 80)
    print("CREATING EXTENDED METRICS COMPARISON")
    print("=" * 80)
    
    models = list(results.keys())
    
    # Collect extended metrics
    metrics_data = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'Specificity': [],
        'F1-Score': [],
        'ROC-AUC': []
    }
    
    for model_name in models:
        metrics_data['Accuracy'].append(results[model_name]['Accuracy'])
        metrics_data['Precision'].append(results[model_name]['Precision'])
        metrics_data['Recall'].append(results[model_name]['Recall'])
        metrics_data['Specificity'].append(results[model_name]['Specificity'])
        metrics_data['F1-Score'].append(results[model_name]['F1-Score'])
        metrics_data['ROC-AUC'].append(results[model_name]['ROC-AUC'])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    x = np.arange(len(models))
    width = 0.13
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    for idx, (metric, values) in enumerate(metrics_data.items()):
        offset = width * idx
        bars = ax.bar(x + offset, values, width, label=metric, 
                      color=colors[idx], alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=7)
    
    ax.set_xlabel('Models', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title('Extended Model Performance Comparison (Including Specificity)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/extended_metrics_comparison.png', dpi=300, bbox_inches='tight')
    
    print("✅ Saved to: results/extended_metrics_comparison.png")


def plot_log_loss_comparison(results):
    """Plot Log Loss comparison for probabilistic models"""
    print("\n" + "=" * 80)
    print("CREATING LOG LOSS COMPARISON")
    print("=" * 80)
    
    # Filter models that have Log Loss
    models_with_logloss = {
        name: metrics['Log Loss'] 
        for name, metrics in results.items() 
        if metrics['Log Loss'] is not None
    }
    
    if not models_with_logloss:
        print("⚠️  No models with probability predictions. Skipping Log Loss plot.")
        return
    
    models = list(models_with_logloss.keys())
    log_losses = list(models_with_logloss.values())
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    bars = ax.bar(models, log_losses, color=colors[:len(models)], 
                  alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, value in zip(bars, log_losses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Log Loss (Lower is Better)', fontweight='bold', fontsize=12)
    ax.set_title('Log Loss Comparison (Probabilistic Performance)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/log_loss_comparison.png', dpi=300, bbox_inches='tight')
    
    print("✅ Saved to: results/log_loss_comparison.png")


def save_results_to_file(results, df):
    """Save comprehensive results to text file"""
    print("\n" + "=" * 80)
    print("SAVING RESULTS TO FILE")
    print("=" * 80)
    
    with open('results/model_comparison.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DROUGHT PREDICTION - COMPREHENSIVE MODEL COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("BASIC METRICS SUMMARY:\n")
        f.write("-" * 80 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        # Extended metrics for each model
        f.write("=" * 80 + "\n")
        f.write("DETAILED EVALUATION FOR EACH MODEL\n")
        f.write("=" * 80 + "\n\n")
        
        for model_name, metrics in results.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"{model_name.upper()}\n")
            f.write(f"{'='*80}\n\n")
            
            # Basic metrics
            f.write("Basic Classification Metrics:\n")
            f.write(f"  Accuracy:     {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)\n")
            f.write(f"  Precision:    {metrics['Precision']:.4f}\n")
            f.write(f"  Recall:       {metrics['Recall']:.4f}\n")
            f.write(f"  Specificity:  {metrics['Specificity']:.4f}\n")
            f.write(f"  F1-Score:     {metrics['F1-Score']:.4f}\n")
            f.write(f"  ROC-AUC:      {metrics['ROC-AUC']:.4f}\n")
            
            # Probabilistic metric
            if metrics['Log Loss'] is not None:
                f.write(f"\nProbabilistic Metric:\n")
                f.write(f"  Log Loss:     {metrics['Log Loss']:.4f}\n")
            
            # Confusion matrix
            cm = metrics['Confusion Matrix']
            tn, fp, fn, tp = cm.ravel()
            f.write(f"\nConfusion Matrix:\n")
            f.write(f"  True Negatives:  {tn:,}\n")
            f.write(f"  False Positives: {fp:,}\n")
            f.write(f"  False Negatives: {fn:,}\n")
            f.write(f"  True Positives:  {tp:,}\n")
            
            # Additional metrics
            f.write(f"\nAdditional Metrics:\n")
            f.write(f"  NPV: {metrics['NPV']:.4f}\n")
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            f.write(f"  False Positive Rate: {fpr:.4f}\n")
            f.write(f"  False Negative Rate: {fnr:.4f}\n")
        
        # Best model summary
        f.write("\n\n" + "=" * 80 + "\n")
        best = max(results.items(), key=lambda x: x[1]['F1-Score'])
        f.write(f"BEST MODEL: {best[0]}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Accuracy:     {best[1]['Accuracy']:.4f}\n")
        f.write(f"Precision:    {best[1]['Precision']:.4f}\n")
        f.write(f"Recall:       {best[1]['Recall']:.4f}\n")
        f.write(f"Specificity:  {best[1]['Specificity']:.4f}\n")
        f.write(f"F1-Score:     {best[1]['F1-Score']:.4f}\n")
        f.write(f"ROC-AUC:      {best[1]['ROC-AUC']:.4f}\n")
        if best[1]['Log Loss'] is not None:
            f.write(f"Log Loss:     {best[1]['Log Loss']:.4f}\n")
    
    print("✅ Saved to: results/model_comparison.txt")