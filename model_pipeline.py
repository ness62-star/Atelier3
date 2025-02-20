import pandas as pd
import numpy as np
from scipy import stats
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import signal
import time
from contextlib import contextmanager
"""from sklearn.preprocessing import LabelEncoder"""
from sklearn.preprocessing import StandardScaler


# Timeout Handler
class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Training timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def prepare_data(train_file, test_file, sample_fraction=1.0):
    """
    Load and prepare data for training and testing.
    - Drops 'Area code' and 'State' columns
    - Encodes 'International plan' and 'Voice mail plan' as binary
    - Handles missing values and removes outliers
    - Normalizes numerical columns
    """
    print("Loading data...")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    # Combine datasets for consistent preprocessing
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"Combined dataset shape: {combined_df.shape}")
    # Drop 'Area code' and 'State' columns
    combined_df.drop(columns=['Area code', 'State'], inplace=True)
    print("Dropped 'Area code' and 'State' columns.")
    # Encode 'International plan' and 'Voice mail plan' as binary
    combined_df['International plan'] = combined_df['International plan'].map({'No': 0, 'Yes': 1})
    combined_df['Voice mail plan'] = combined_df['Voice mail plan'].map({'No': 0, 'Yes': 1})
    print("Encoded 'International plan' and 'Voice mail plan' as binary.")
    # Drop rows with more than 50% missing values
    combined_df = combined_df.dropna(thresh=combined_df.shape[1] * 0.5)
    print("Dropped rows with more than 50% missing values.")
    # Fill missing values with mean (numeric columns only)
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    combined_df[numeric_cols] = combined_df[numeric_cols].fillna(combined_df[numeric_cols].mean())
    # Data Normalization and Scaling
    scaler = StandardScaler()
    combined_df[numeric_cols] = scaler.fit_transform(combined_df[numeric_cols])
    print("Normalized and scaled numerical columns.")
    # Remove Outliers using Z-score
    z_scores = np.abs(stats.zscore(combined_df[numeric_cols]))
    combined_df = combined_df[(z_scores < 3).all(axis=1)]
    print("Removed outliers using Z-score.")
    # Split back to training and testing datasets
    train_size = len(train_df)
    train_df = combined_df.iloc[:train_size]
    test_df = combined_df.iloc[train_size:]
    # Separate features and target
    target = 'Churn'
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target].astype(int)
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target].astype(int)

    print("Data preparation completed.")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, max_depth=3, min_samples_split=20, min_samples_leaf=10):
    """
    Train a Decision Tree with controlled depth and minimum samples.
    Added detailed logging to track progress.
    """
    print("Initializing Decision Tree Classifier...")
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    print("Starting model training...")
    start_time = time.time()
    # Track the training progress
    print("Fitting model...")
    model.fit(X_train, y_train)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Model training completed in {duration:.2f} seconds.")
    return model


def save_model(model, filename="decision_tree_model.pkl"):
    """
    Sauvegarde le modèle entraîné dans un fichier.
    """
    joblib.dump(model, filename)
    print(f"Modèle sauvegardé sous {filename}")


def load_model(filename="decision_tree_model.pkl"):
    """
    Charge un modèle sauvegardé.
    """
    return joblib.load(filename)


def evaluate_model(model, X_test, y_test):
    """Évaluer le modèle et afficher les métriques de performance."""
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Retourner la précision pour qu'elle soit utilisée dans le main.py
    return accuracy
