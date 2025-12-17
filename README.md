# 🌵 Drought Prediction – Machine Learning Classification Project

**ADDIS ABABA INSTITUTE OF TECHNOLOGY**  
Department of Software Engineering

---

## 📋 Project Title
**Predicting Drought Conditions Using Multi-Algorithm Classification**

---

## 👥 Team Members

| Name | ID |
|------|-----|
| [Yeabsira Samuel] | [ATE/9305/14] |
| [Natnael Nigatu] | [ATE/7495/14] |
| [Kassahun Belachew] | [ATE/8400/14] |
| [Tsegaab Alemu] | [ATE/8814/14] |

---

## 📅 Submission Details

- **Date:** December 17 20245
- **Submitted to:** [Mr. Bisrat]
- **Course:** Machine Learning 

---

## 📊 Project Overview

This project implements a **comprehensive machine learning classification system** using **4 state-of-the-art algorithms** to predict drought conditions based on meteorological and spatial data. The system demonstrates advanced understanding of classification techniques, extensive evaluation methodologies, and production-ready implementation.

### 🎯 Objective
Predict binary drought conditions (Drought / No Drought) using 14 meteorological features, achieving **99.70% accuracy** with the best model.

### 🔬 Algorithms Implemented
1. **Logistic Regression** (Newton's Method optimization)
2. **Support Vector Machine** (RBF kernel)
3. **K-Nearest Neighbors** (K=5)
4. **Gaussian Naive Bayes** (Probabilistic classifier)

---

## 🏆 Key Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
|-------|----------|-----------|--------|----------|---------|---------------|
| **🥇 Logistic Regression** | **99.70%** | **99.43%** | **99.99%** | **99.71%** | **1.0000** | 0.36s |
| 🥈 SVM | 99.24% | 98.63% | 99.91% | 99.27% | 0.9999 | 253.11s |
| 🥉 KNN | 95.68% | 94.40% | 97.44% | 95.90% | 0.9910 | 0.28s |
| Naive Bayes | 94.66% | 91.70% | 98.60% | 95.02% | 0.9936 | 0.04s |

### 🎖️ Best Model: Logistic Regression
- ✅ **99.70% Accuracy** – Correctly predicts 23,127 out of 23,197 test samples
- ✅ **99.99% Recall** – Misses only 1 drought out of 12,000 (critical for early warning!)
- ✅ **0.0174 Log Loss** – Excellent probabilistic calibration
- ✅ **Perfect ROC-AUC (1.0000)** – Flawless class separation
- ✅ **Fast Training** – Only 0.36 seconds

### 📈 Dataset Statistics
- **Total Samples:** 115,985
- **Features:** 14 meteorological and spatial indicators
- **Target Distribution:** 48.3% No Drought | 51.7% Drought (Balanced)
- **Train/Val/Test Split:** 70% / 10% / 20%

---

## 🔍 Features Used

### 🌡️ Temperature & Humidity (4 features)
1. `RH2M` – Relative Humidity at 2 Meters (%)
2. `T2M_MAX` – Maximum Temperature (°C)
3. `T2M_MIN` – Minimum Temperature (°C)
4. `T2M` – Mean Temperature (°C)

### 💨 Wind & Solar Radiation (2 features)
5. `WS2M` – Wind Speed at 2 Meters (m/s)
6. `ALLSKY_SFC_SW_DWN` – Solar Radiation (kW-hr/m²/day)

### 🌧️ Precipitation & Drought Index (2 features)
7. `PRECTOTCORR` – Precipitation (mm/day)
8. `spei` – **SPEI Drought Index** (-3 to +3, most important feature)

### 🗺️ Spatial & Temporal (6 features)
9-14. `lat_sin`, `lat_cos`, `lon_sin`, `lon_cos`, `month_sin`, `month_cos`  
    *(Trigonometric encoding of geographic and seasonal information)*

---

## 📊 Evaluation Metrics (15+ Methods)

### Prediction-Based Metrics
- ✅ Confusion Matrix (TP, TN, FP, FN breakdown)
- ✅ Accuracy
- ✅ Precision
- ✅ Recall (Sensitivity)
- ✅ Specificity
- ✅ F1-Score

### Probabilistic Metrics
- ✅ Log Loss
- ✅ NPV (Negative Predictive Value)
- ✅ False Positive Rate (FPR)
- ✅ False Negative Rate (FNR)

### Threshold-Independent
- ✅ ROC-AUC Score
- ✅ ROC Curves (Visual)
- ✅ Precision-Recall Curves

### Comprehensive Reporting
- ✅ Per-Class Performance Analysis
- ✅ Classification Reports

---

## 🎨 Visualizations (9 Professional Plots)

### Basic Visualizations
1. **Confusion Matrices** (2×2 grid for all models)
2. **Metrics Comparison** (Bar chart, 5 metrics)
3. **Extended Metrics** (Including Specificity)
4. **Log Loss Comparison** (Probabilistic performance)

### Advanced Visualizations
5. **ROC Curves** (All models on one plot)
6. **Precision-Recall Curves** (Threshold analysis)
7. **Performance Radar Chart** (Spider chart, 5 metrics)
8. **Performance Heatmap** (Color-coded comparison)
9. **Error Analysis** (False Positives vs False Negatives)

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- pip package manager
- 4GB RAM minimum
- 1GB free disk space

### Setup Instructions
```bash
# 1. Clone the repository
git clone <your-repo-url>
cd drought_prediction_project

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python --version  # Should be 3.10+
```

---

## 📖 Usage Guide

### Step 1: Train All Models
```bash
python main.py
```

**What happens during training:**
1. ✅ Data preprocessing (loading, splitting, scaling)
2. ✅ Training 4 algorithms with optimal hyperparameters
3. ✅ Comprehensive evaluation with 15+ metrics
4. ✅ Generation of 9 professional visualizations
5. ✅ Saving models, scaler, and results

**Outputs:**
- `models/` – 4 trained models + scaler (.pkl files)
- `results/` – 9 visualizations (.png files)
- `results/model_comparison.txt` – Comprehensive report

**Execution Time:** ~7-8 minutes (SVM takes ~6 minutes)

---

### Step 2: Make Predictions on New Data
```bash
python predict_new_data.py
```

**Interactive Prediction System Features:**

#### 🎮 6 Prediction Options:

1. **🖊️ Manual Input** – Enter all 14 feature values step-by-step
2. **🌵 Drought Example** – Test with severe drought conditions
3. **💧 No Drought Example** – Test with normal wet conditions
4. **⚠️ Moderate Risk Example** – Test borderline conditions
5. **📋 Feature Information** – View detailed feature descriptions
6. **❌ Exit** – Close the system

#### 🎯 Prediction Output:
- ✅ Individual predictions from all 4 models
- ✅ Confidence percentages (when available)
- ✅ Majority vote consensus
- ✅ Color-coded results (🌵 Drought / 💧 No Drought)
- ✅ Actionable recommendations

---

## 📁 Project Structure
```
drought_prediction_project/
│
├── data/
│   └── raw/
│       └── stage_4_drought_dataset.csv    # Dataset (115,985 samples)
│
├── models/
│   ├── logistic_regression.pkl             # Trained models
│   ├── svm.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   └── scaler.pkl                          # Feature scaler
│
├── results/
│   ├── confusion_matrices.png              # 9 visualizations
│   ├── metrics_comparison.png
│   ├── extended_metrics_comparison.png
│   ├── log_loss_comparison.png
│   ├── roc_curves.png
│   ├── precision_recall_curves.png
│   ├── performance_radar.png
│   ├── performance_heatmap.png
│   ├── error_analysis.png
│   └── model_comparison.txt                # Comprehensive report
│
├── src/
│   ├── __init__.py
│   ├── utils.py                            # Helper functions
│   ├── data_preprocessing.py               # Data pipeline
│   ├── train_models.py                     # Training logic
│   ├── evaluate_models.py                  # Evaluation & metrics
│   ├── visualizations.py                   # Advanced plots
│   └── predict.py                          # Prediction functions
│
├── main.py                                  # Main execution script
├── predict_new_data.py                     # Interactive prediction CLI
├── requirements.txt                         # Python dependencies
└── README.md                               # This file
```

---

## 🎓 Machine Learning Theory

### Algorithms Explained

#### 1. 🔵 Logistic Regression (Newton's Method)

**Type:** Linear classifier with probabilistic output

**Equation:**
```
P(Drought|X) = 1 / (1 + e^(-(β₀ + β₁x₁ + ... + β₁₄x₁₄)))
```

**Optimization:** Newton-Raphson method
- Faster convergence than gradient descent
- Uses second-order derivatives (Hessian matrix)

**Advantages:**
- ✅ Interpretable coefficients
- ✅ Probabilistic predictions
- ✅ Fast training & prediction
- ✅ Works well on linearly separable data

**Our Results:** 99.70% accuracy, 0.36s training time

---

#### 2. 🟣 Support Vector Machine (SVM)

**Type:** Maximum-margin classifier

**Kernel:** Radial Basis Function (RBF)
```
K(x, x') = e^(-γ||x - x'||²)
```

**Objective:** Find hyperplane that maximizes margin between classes

**Advantages:**
- ✅ Effective in high-dimensional spaces
- ✅ Robust to outliers
- ✅ Excellent for complex decision boundaries

**Our Results:** 99.24% accuracy, but slow training (253s)

---

#### 3. 🟢 K-Nearest Neighbors (KNN)

**Type:** Instance-based (lazy) learner

**Decision Rule:**
```
Class = Majority vote of K=5 nearest neighbors
```

**Distance Metric:** Euclidean distance in scaled feature space

**Advantages:**
- ✅ No training phase (instant)
- ✅ Naturally handles non-linear boundaries
- ✅ Intuitive and simple

**Our Results:** 95.68% accuracy, 0.28s training time

---

#### 4. 🟡 Gaussian Naive Bayes

**Type:** Probabilistic classifier

**Bayes' Theorem:**
```
P(Drought|X) = P(X|Drought) × P(Drought) / P(X)
```

**Assumption:** Features are conditionally independent given the class

**Advantages:**
- ✅ Extremely fast training (0.04s)
- ✅ Works well with small datasets
- ✅ Handles noisy data gracefully

**Our Results:** 94.66% accuracy, fastest training

---

## 🔧 Feature Engineering

### Preprocessing Pipeline

1. **Feature Extraction**
   - Remove non-predictive columns (row_id)
   - Separate features (X) and target (y)

2. **Train-Test Split**
   - 70% Training (81,189 samples)
   - 10% Validation (11,599 samples)
   - 20% Testing (23,197 samples)
   - Stratified split maintains class balance

3. **Feature Scaling**
   - Method: StandardScaler (Z-score normalization)
   - Formula: `z = (x - μ) / σ`
   - Applied only on training data, then transformed to val/test
   - Saved for production use

**Why Scaling?**
- ✅ Improves convergence speed
- ✅ Prevents feature dominance
- ✅ Essential for distance-based algorithms (KNN, SVM)

---

## 🧪 Validation Strategy

### Evaluation Approach
```
Training Set (70%)
    ↓
Train Models
    ↓
Validation Set (10%)  →  Hyperparameter tuning (if needed)
    ↓
Test Set (20%)  →  Final evaluation (never seen during training)
```

**Why This Matters:**
- Prevents overfitting
- Ensures generalization to unseen data
- Provides unbiased performance estimates

---

## 💡 Real-World Applications

### 🌍 Agricultural Planning
- Early drought warnings for farmers
- Irrigation scheduling optimization
- Crop selection recommendations

### 🏛️ Government Policy
- Resource allocation for drought relief
- Emergency preparedness planning
- Climate adaptation strategies

### 💧 Water Management
- Reservoir management
- Water rationing decisions
- Drought severity monitoring

### 📊 Climate Research
- Drought pattern analysis
- Climate change impact studies
- Seasonal forecasting

---

## 🔍 Key Insights from Results

### 1. Feature Importance
**SPEI (Drought Index) is the dominant predictor**
- Single most important feature
- Strong negative correlation (-0.79) with drought
- Scientifically validated drought indicator

### 2. Model Comparison
**Logistic Regression is the optimal choice:**
- ✅ Highest accuracy (99.70%)
- ✅ Perfect recall (99.99%) – Critical for early warning!
- ✅ Fast training (0.36s)
- ✅ Interpretable coefficients

**SVM is a strong alternative:**
- ✅ Second-best accuracy (99.24%)
- ⚠️ Very slow training (253s)
- ✅ Excellent for production if trained offline

### 3. Error Analysis
**Logistic Regression error breakdown:**
- False Positives: 69 (predicted drought, actually normal)
- False Negatives: 1 (predicted normal, actually drought)
- **Implication:** System errs on the side of caution (good for disaster prevention!)

---

## 📚 Learning Outcomes

This project demonstrates comprehensive understanding of:

### Machine Learning Fundamentals
✅ Classification problem formulation  
✅ Train-test-validation splitting  
✅ Feature scaling and preprocessing  
✅ Model selection and comparison  

### Algorithm Implementation
✅ Logistic Regression (Newton's Method)  
✅ Support Vector Machines (RBF kernel)  
✅ K-Nearest Neighbors (instance-based)  
✅ Naive Bayes (probabilistic)  

### Evaluation & Validation
✅ 15+ evaluation metrics  
✅ Confusion matrix analysis  
✅ ROC and Precision-Recall curves  
✅ Cross-model comparison  

### Software Engineering
✅ Modular code architecture  
✅ Production-ready pipeline  
✅ Model serialization (pickle)  
✅ Interactive user interface  
✅ Comprehensive documentation  

### Data Visualization
✅ 9 professional visualizations  
✅ Matplotlib and Seaborn mastery  
✅ Statistical plot interpretation  

---

## 🎯 Future Enhancements

### Potential Improvements
- [ ] Deep Learning models (LSTM for temporal patterns)
- [ ] Real-time API deployment (Flask/FastAPI)
- [ ] Hyperparameter optimization (Grid Search / Bayesian)
- [ ] Feature selection using Random Forest importance
- [ ] Ensemble methods (Voting Classifier, Stacking)
- [ ] Web dashboard for visualization (Streamlit/Dash)
- [ ] Mobile app integration
- [ ] Real-time satellite data integration

---

## 📦 Dependencies
```txt
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
joblib==1.3.2
```

**Install all:**
```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request


---

## 🙏 Acknowledgments

### Data Source
- **Stage 4 Drought Dataset** – [Source information]
- **SPEI Calculation** – Based on standardized meteorological indices


---


## 📊 Performance Summary Table

| Metric | Logistic Regression | SVM | KNN | Naive Bayes |
|--------|---------------------|-----|-----|-------------|
| **Accuracy** | 99.70% | 99.24% | 95.68% | 94.66% |
| **Precision** | 99.43% | 98.63% | 94.40% | 91.70% |
| **Recall** | 99.99% | 99.91% | 97.44% | 98.60% |
| **F1-Score** | 99.71% | 99.27% | 95.90% | 95.02% |
| **ROC-AUC** | 1.0000 | 0.9999 | 0.9910 | 0.9936 |
| **Log Loss** | 0.0174 | 0.0130 | 0.1904 | 0.1530 |
| **Training Time** | 0.36s | 253.11s | 0.28s | 0.04s |
| **Errors (Total)** | 70 | 177 | 1,001 | 1,239 |

---

## 🎉 Project Completion

**Status:** ✅ Complete  
**Total Lines of Code:** ~2,500+  
**Total Execution Time:** ~8 minutes  

---

<div align="center">

### ⭐ Star this repository if you found it helpful!

---

*Predicting Droughts, Saving Lives* 🌵💧

</div>
