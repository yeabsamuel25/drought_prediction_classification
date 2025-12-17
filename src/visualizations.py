"""
Advanced Visualizations Module
Creates ROC curves, Precision-Recall curves, and other advanced plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def plot_roc_curves(results, X_test, y_test, models):
    """Plot ROC curves for all models"""
    print("\n" + "=" * 80)
    print("CREATING ROC CURVES")
    print("=" * 80)
    
    plt.figure(figsize=(10, 8))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    for idx, (model_name, model) in enumerate(models.items()):
        # Get probability predictions
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.predict(X_test)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2.5,
                label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5000)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
    plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/roc_curves.png', dpi=300, bbox_inches='tight')
    
    print("✅ Saved to: results/roc_curves.png")


def plot_precision_recall_curves(results, X_test, y_test, models):
    """Plot Precision-Recall curves for all models"""
    print("\n" + "=" * 80)
    print("CREATING PRECISION-RECALL CURVES")
    print("=" * 80)
    
    plt.figure(figsize=(10, 8))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    for idx, (model_name, model) in enumerate(models.items()):
        # Get probability predictions
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.predict(X_test)
        
        # Calculate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        
        # Calculate average precision
        avg_precision = average_precision_score(y_test, y_proba)
        
        # Plot
        plt.plot(recall, precision, color=colors[idx % len(colors)], lw=2.5,
                label=f'{model_name} (AP = {avg_precision:.4f})')
    
    # Add baseline
    baseline = (y_test == 1).sum() / len(y_test)
    plt.axhline(y=baseline, color='k', linestyle='--', lw=2, 
                label=f'Baseline (AP = {baseline:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curves - All Models', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/precision_recall_curves.png', dpi=300, bbox_inches='tight')
    
    print("✅ Saved to: results/precision_recall_curves.png")


def plot_performance_radar(results):
    """Create radar chart comparing model performance"""
    print("\n" + "=" * 80)
    print("CREATING PERFORMANCE RADAR CHART")
    print("=" * 80)
    
    # Metrics to compare
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    # Number of variables
    num_vars = len(metrics)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    # Plot each model
    for idx, (model_name, model_results) in enumerate(results.items()):
        values = [model_results[metric] for metric in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=model_name, color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
    
    # Fix axis to go in the right order
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True)
    
    plt.title('Model Performance Radar Chart', 
             fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/performance_radar.png', dpi=300, bbox_inches='tight')
    
    print("✅ Saved to: results/performance_radar.png")


def plot_model_comparison_heatmap(results):
    """Create heatmap of model performance across metrics"""
    print("\n" + "=" * 80)
    print("CREATING PERFORMANCE HEATMAP")
    print("=" * 80)
    
    # Prepare data
    metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'ROC-AUC']
    models = list(results.keys())
    
    data = []
    for model_name in models:
        row = [results[model_name][metric] for metric in metrics]
        data.append(row)
    
    df = pd.DataFrame(data, index=models, columns=metrics)
    
    # Create heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt='.4f', cmap='RdYlGn', 
                cbar_kws={'label': 'Score'}, vmin=0.9, vmax=1.0,
                linewidths=0.5, linecolor='gray')
    
    plt.title('Model Performance Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Models', fontweight='bold', fontsize=12)
    plt.xlabel('Metrics', fontweight='bold', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('results/performance_heatmap.png', dpi=300, bbox_inches='tight')
    
    print("✅ Saved to: results/performance_heatmap.png")


def plot_error_analysis(results):
    """Plot error analysis comparing false positives and false negatives"""
    print("\n" + "=" * 80)
    print("CREATING ERROR ANALYSIS")
    print("=" * 80)
    
    models = list(results.keys())
    
    false_positives = []
    false_negatives = []
    
    for model_name in models:
        cm = results[model_name]['Confusion Matrix']
        tn, fp, fn, tp = cm.ravel()
        false_positives.append(fp)
        false_negatives.append(fn)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    # False Positives
    bars1 = ax1.bar(models, false_positives, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Count', fontweight='bold', fontsize=12)
    ax1.set_title('False Positives by Model\n(Predicted Drought, Actually No Drought)', 
                  fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars1, false_positives):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(value):,}', ha='center', va='bottom', fontweight='bold')
    
    # False Negatives
    bars2 = ax2.bar(models, false_negatives, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Count', fontweight='bold', fontsize=12)
    ax2.set_title('False Negatives by Model\n(Predicted No Drought, Actually Drought)', 
                  fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars2, false_negatives):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(value):,}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Error Analysis - All Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/error_analysis.png', dpi=300, bbox_inches='tight')
    
    print("✅ Saved to: results/error_analysis.png")