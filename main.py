import sys
import os
import time

# Add src to path
sys.path.append('src')

from data_preprocessing import preprocess_pipeline
from train_models import train_all_models
from evaluate_models import (
    evaluate_all_models,
    print_results,
    print_extended_results,
    plot_confusion_matrices,
    plot_metrics_comparison,
    plot_extended_metrics,
    plot_log_loss_comparison,
    save_results_to_file
)
from visualizations import (
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_performance_radar,
    plot_model_comparison_heatmap,
    plot_error_analysis
)
from utils import create_directories, print_header, format_time


def print_welcome_banner():
    """Print welcome banner with project information"""
    print("\n" + "🌵" * 40)
    print("=" * 80)
    print("        DROUGHT PREDICTION CLASSIFICATION PROJECT")
    print("        Comprehensive Machine Learning Analysis")
    print("=" * 80)
    print("🌵" * 40)
    
    print("\n📊 PROJECT OVERVIEW:")
    print("   Dataset: Stage 4 Drought Dataset (115,985 samples)")
    print("   Features: 14 meteorological and spatial indicators")
    print("   Target: Binary classification (Drought / No Drought)")
    print("   Split: 70% Train / 10% Validation / 20% Test")
    
    print("\n🤖 TRAINING 4 ALGORITHMS:")
    print("   1. Logistic Regression (Newton's Method)")
    print("   2. Support Vector Machine (RBF Kernel)")
    print("   3. K-Nearest Neighbors (K=5)")
    print("   4. Gaussian Naive Bayes")
    
    print("\n📈 EVALUATION METRICS (15+ Methods):")
    print("   • Prediction-Based: Confusion Matrix, Accuracy, Precision, Recall, Specificity, F1")
    print("   • Probabilistic: Log Loss, NPV")
    print("   • Threshold-Independent: ROC-AUC, ROC Curves, Precision-Recall Curves")
    print("   • Error Analysis: FPR, FNR, Per-Class Performance")
    
    print("\n🎨 VISUALIZATIONS (9+ Plots):")
    print("   • Confusion Matrices, Metrics Comparison, Extended Metrics")
    print("   • Log Loss Comparison, ROC Curves, Precision-Recall Curves")
    print("   • Performance Radar, Performance Heatmap, Error Analysis")
    
    print("\n" + "=" * 80)


def print_phase_summary(phase_name, duration=None):
    """Print phase completion summary"""
    print("\n" + "✅" * 40)
    print(f"✅ {phase_name} COMPLETED!")
    if duration:
        print(f"⏱️  Duration: {format_time(duration)}")
    print("✅" * 40)


def main():
    """Main execution function"""
    total_start_time = time.time()
    
    # Welcome Banner
    print_welcome_banner()
    
    # Setup
    print("\n🔧 Setting up directories...")
    create_directories()
    print("✅ Directories verified")
    
    # ========================================================================
    # PHASE 1: DATA PREPROCESSING
    # ========================================================================
    print_header("PHASE 1: DATA PREPROCESSING")
    phase1_start = time.time()
    
    dataset_path = 'data/raw/stage_4_drought_dataset.csv'
    
    if not os.path.exists(dataset_path):
        print(f"\n❌ ERROR: Dataset not found!")
        print(f"   Expected location: {dataset_path}")
        print(f"   Please place 'stage_4_drought_dataset.csv' in: data/raw/")
        return
    
    try:
        data = preprocess_pipeline(dataset_path)
        
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        
        print_phase_summary("DATA PREPROCESSING", time.time() - phase1_start)
        
    except Exception as e:
        print(f"\n❌ ERROR in preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # PHASE 2: MODEL TRAINING
    # ========================================================================
    print_header("PHASE 2: MODEL TRAINING")
    phase2_start = time.time()
    
    try:
        models, training_times = train_all_models(X_train, y_train)
        
        print_phase_summary("MODEL TRAINING", time.time() - phase2_start)
        
    except Exception as e:
        print(f"\n❌ ERROR in training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # PHASE 3: MODEL EVALUATION
    # ========================================================================
    print_header("PHASE 3: MODEL EVALUATION")
    phase3_start = time.time()
    
    try:
        results = evaluate_all_models(models, X_test, y_test)
        
        # Print basic results
        df_comparison = print_results(results)
        
        # Print extended results with all metrics
        print_extended_results(results)
        
        print_phase_summary("MODEL EVALUATION", time.time() - phase3_start)
        
    except Exception as e:
        print(f"\n❌ ERROR in evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # PHASE 4: BASIC VISUALIZATIONS
    # ========================================================================
    print_header("PHASE 4: CREATING BASIC VISUALIZATIONS")
    phase4_start = time.time()
    
    try:
        plot_confusion_matrices(results)
        plot_metrics_comparison(results)
        plot_extended_metrics(results)
        plot_log_loss_comparison(results)
        
        print_phase_summary("BASIC VISUALIZATIONS", time.time() - phase4_start)
        
    except Exception as e:
        print(f"\n⚠️  Warning in basic visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # PHASE 5: ADVANCED VISUALIZATIONS
    # ========================================================================
    print_header("PHASE 5: CREATING ADVANCED VISUALIZATIONS")
    phase5_start = time.time()
    
    try:
        plot_roc_curves(results, X_test, y_test, models)
        plot_precision_recall_curves(results, X_test, y_test, models)
        plot_performance_radar(results)
        plot_model_comparison_heatmap(results)
        plot_error_analysis(results)
        
        print_phase_summary("ADVANCED VISUALIZATIONS", time.time() - phase5_start)
        
    except Exception as e:
        print(f"\n⚠️  Warning in advanced visualizations: {e}")
        print("     Continuing with remaining tasks...")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # PHASE 6: SAVE COMPREHENSIVE RESULTS
    # ========================================================================
    print_header("PHASE 6: SAVING RESULTS")
    
    try:
        save_results_to_file(results, df_comparison)
        print("\n✅ Results saved to file")
    except Exception as e:
        print(f"\n⚠️  Warning in saving results: {e}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    total_time = time.time() - total_start_time
    
    print("\n" + "=" * 80)
    print("=" * 80)
    print("                    🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    print("=" * 80)
    print("=" * 80)
    
    print(f"\n⏱️  TOTAL EXECUTION TIME: {format_time(total_time)}")
    
    # Training Time Breakdown
    print("\n📊 TRAINING TIME BREAKDOWN:")
    print("-" * 80)
    for model_name, train_time in training_times.items():
        time_str = f"{train_time:.2f}s" if train_time < 60 else f"{train_time/60:.2f}m"
        print(f"   {model_name:.<35} {time_str:>10}")
    total_train_time = sum(training_times.values())
    print("-" * 80)
    time_str = f"{total_train_time:.2f}s" if total_train_time < 60 else f"{total_train_time/60:.2f}m"
    print(f"   {'TOTAL TRAINING TIME':.<35} {time_str:>10}")
    
    # Files Saved
    print("\n📂 FILES SAVED:")
    print("=" * 80)
    
    print("\n   📊 TRAINED MODELS (models/):")
    print("   ├── logistic_regression.pkl")
    print("   ├── svm.pkl")
    print("   ├── knn.pkl")
    print("   └── naive_bayes.pkl")
    
    print("\n   📈 BASIC VISUALIZATIONS (results/):")
    print("   ├── confusion_matrices.png          (2×2 grid)")
    print("   ├── metrics_comparison.png          (5 metrics)")
    print("   ├── extended_metrics_comparison.png (6 metrics + Specificity)")
    print("   └── log_loss_comparison.png         (Probabilistic)")
    
    print("\n   🎨 ADVANCED VISUALIZATIONS (results/):")
    print("   ├── roc_curves.png                  (ROC curves for all models)")
    print("   ├── precision_recall_curves.png     (PR curves for all models)")
    print("   ├── performance_radar.png           (Spider/Radar chart)")
    print("   ├── performance_heatmap.png         (Color-coded table)")
    print("   └── error_analysis.png              (FP vs FN comparison)")
    
    print("\n   📄 COMPREHENSIVE REPORT (results/):")
    print("   └── model_comparison.txt            (Full evaluation details)")
    
    # Best Model Summary
    print("\n" + "🏆" * 40)
    print("                         BEST MODEL SUMMARY")
    print("🏆" * 40)
    
    best = max(results.items(), key=lambda x: x[1]['F1-Score'])
    
    print(f"\n   Model: {best[0]}")
    print("   " + "-" * 60)
    print(f"   Accuracy:     {best[1]['Accuracy']:.4f} ({best[1]['Accuracy']*100:.2f}%)")
    print(f"   Precision:    {best[1]['Precision']:.4f}")
    print(f"   Recall:       {best[1]['Recall']:.4f}")
    print(f"   Specificity:  {best[1]['Specificity']:.4f}")
    print(f"   F1-Score:     {best[1]['F1-Score']:.4f}")
    print(f"   ROC-AUC:      {best[1]['ROC-AUC']:.4f}")
    if best[1]['Log Loss'] is not None:
        print(f"   Log Loss:     {best[1]['Log Loss']:.4f}")
    
    # Confusion Matrix Summary
    cm = best[1]['Confusion Matrix']
    tn, fp, fn, tp = cm.ravel()
    print("\n   Confusion Matrix:")
    print(f"   ├── True Negatives:  {tn:,}")
    print(f"   ├── False Positives: {fp:,}")
    print(f"   ├── False Negatives: {fn:,}")
    print(f"   └── True Positives:  {tp:,}")
    
    print("\n" + "🏆" * 40)
    
    # Performance Ranking
    print("\n📊 MODEL PERFORMANCE RANKING (by F1-Score):")
    print("=" * 80)
    
    ranked = sorted(results.items(), key=lambda x: x[1]['F1-Score'], reverse=True)
    
    medals = ['🥇', '🥈', '🥉', '4️⃣']
    for idx, (model_name, metrics) in enumerate(ranked):
        medal = medals[idx] if idx < len(medals) else f"{idx+1}."
        print(f"   {medal} {model_name:.<30} "
              f"F1: {metrics['F1-Score']:.4f} | "
              f"Acc: {metrics['Accuracy']:.4f} | "
              f"AUC: {metrics['ROC-AUC']:.4f}")
    
    # Key Insights
    print("\n💡 KEY INSIGHTS:")
    print("=" * 80)
    
    # Find model with fewest false negatives (most important for drought!)
    best_recall = max(results.items(), key=lambda x: x[1]['Recall'])
    print(f"   • Best at Catching Droughts: {best_recall[0]} "
          f"(Recall: {best_recall[1]['Recall']:.4f})")
    
    # Find fastest model
    fastest = min(training_times.items(), key=lambda x: x[1])
    print(f"   • Fastest Training: {fastest[0]} ({fastest[1]:.2f}s)")
    
    # Find most accurate
    best_acc = max(results.items(), key=lambda x: x[1]['Accuracy'])
    print(f"   • Highest Accuracy: {best_acc[0]} "
          f"({best_acc[1]['Accuracy']*100:.2f}%)")
    
    # Total errors
    total_samples = len(y_test)
    best_cm = best[1]['Confusion Matrix']
    total_errors = best_cm[0,1] + best_cm[1,0]  # FP + FN
    print(f"   • Total Errors (Best Model): {total_errors:,} / {total_samples:,} "
          f"({total_errors/total_samples*100:.2f}%)")
    
    # Next Steps
    print("\n🚀 NEXT STEPS:")
    print("=" * 80)
    print("   1. Review visualizations in results/ folder")
    print("   2. Read comprehensive report: results/model_comparison.txt")
    print("   3. Test predictions: python predict_new_data.py (if available)")
    print("   4. Deploy best model for production use")
    
    print("\n" + "=" * 80)
    print("                    Thank you for using this system!")
    print("                  Project completed successfully! 🎉")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user (Ctrl+C)")
        print("Exiting gracefully...")
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ CRITICAL ERROR")
        print("=" * 80)
        print(f"\nError: {e}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 80)