"""
Model Training Module
Trains 4 classification algorithms
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import joblib
import time


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression with Newton's Method"""
    print("\n" + "=" * 80)
    print("🔵 TRAINING: LOGISTIC REGRESSION (Newton's Method)")
    print("=" * 80)
    
    model = LogisticRegression(
        solver='newton-cg',  # Newton's Method
        max_iter=1000,
        random_state=42
    )
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    print(f"✅ Trained in {train_time:.2f} seconds")
    
    return model, train_time


def train_svm(X_train, y_train):
    """Train Support Vector Machine"""
    print("\n" + "=" * 80)
    print("🟣 TRAINING: SUPPORT VECTOR MACHINE (SVM)")
    print("=" * 80)
    
    model = SVC(
        kernel='rbf',
        C=1.0,
        probability=True,  # Needed for ROC-AUC
        random_state=42
    )
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    print(f"✅ Trained in {train_time:.2f} seconds")
    print(f"   Support Vectors: {model.n_support_}")
    
    return model, train_time


def train_knn(X_train, y_train):
    """Train K-Nearest Neighbors"""
    print("\n" + "=" * 80)
    print("🟢 TRAINING: K-NEAREST NEIGHBORS (KNN)")
    print("=" * 80)
    
    model = KNeighborsClassifier(
        n_neighbors=5,
        weights='uniform'
    )
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    print(f"✅ Trained in {train_time:.2f} seconds")
    
    return model, train_time


def train_naive_bayes(X_train, y_train):
    """Train Gaussian Naive Bayes"""
    print("\n" + "=" * 80)
    print("🟡 TRAINING: GAUSSIAN NAIVE BAYES")
    print("=" * 80)
    
    model = GaussianNB()
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    print(f"✅ Trained in {train_time:.2f} seconds")
    
    return model, train_time


def train_all_models(X_train, y_train):
    """Train all 4 models"""
    print("\n" + "🚀" * 40)
    print("TRAINING ALL MODELS")
    print("🚀" * 40)
    
    models = {}
    times = {}
    
    # Train each model
    models['Logistic Regression'], times['Logistic Regression'] = train_logistic_regression(X_train, y_train)
    models['SVM'], times['SVM'] = train_svm(X_train, y_train)
    models['KNN'], times['KNN'] = train_knn(X_train, y_train)
    models['Naive Bayes'], times['Naive Bayes'] = train_naive_bayes(X_train, y_train)
    
    # Save models
    print("\n" + "=" * 80)
    print("SAVING MODELS")
    print("=" * 80)
    
    joblib.dump(models['Logistic Regression'], 'models/logistic_regression.pkl')
    joblib.dump(models['SVM'], 'models/svm.pkl')
    joblib.dump(models['KNN'], 'models/knn.pkl')
    joblib.dump(models['Naive Bayes'], 'models/naive_bayes.pkl')
    
    print("✅ All models saved to models/ folder")
    
    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    for name, t in times.items():
        print(f"   {name:.<30} {t:.2f}s")
    print(f"\n   {'TOTAL':.<30} {sum(times.values()):.2f}s")
    
    print("\n" + "✅" * 40)
    print("ALL MODELS TRAINED!")
    print("✅" * 40)
    
    return models, times