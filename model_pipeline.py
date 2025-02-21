import pandas as pd
import numpy as np
"""from scipy import stats"""
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import signal
import time
from contextlib import contextmanager
from sklearn.preprocessing import StandardScaler
""", confusion_matrix, ConfusionMatrixDisplay"""
"""import matplotlib.pyplot as plt"""



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
    """
    print("Loading data...")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    # Apply sample fraction to training data only
    if sample_fraction < 1.0:
        train_df = train_df.sample(frac=sample_fraction, random_state=42)
        print(f"Using {sample_fraction*100}% of training data")
    # Essential preprocessing only
    for df in [train_df, test_df]:
        # Drop unnecessary columns
        df.drop(columns=['Area code', 'State'], inplace=True)
        # Binary encoding
        df['International plan'] = df['International plan'].map({'No': 0, 'Yes': 1})
        df['Voice mail plan'] = df['Voice mail plan'].map({'No': 0, 'Yes': 1})
        # Handle missing values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.drop('Churn')
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])
    # Separate features and target
    X_train = train_df.drop(columns=['Churn'])
    y_train = train_df['Churn'].astype(int)
    X_test = test_df.drop(columns=['Churn'])
    y_test = test_df['Churn'].astype(int)
    print("Data preparation completed.")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, max_depth=3, min_samples_split=20, min_samples_leaf=10, timeout=300):
    """Train a Decision Tree with timeout"""
    print("Training model...")
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    start_time = time.time()
    try:
        with time_limit(timeout):
            model.fit(X_train, y_train)
    except TimeoutException:
        raise TimeoutError(f"Training exceeded {timeout} seconds limit")
    duration = time.time() - start_time
    print(f"Training completed in {duration:.2f} seconds")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and display performance metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\nModel Evaluation:")
    print(f'Accuracy: {accuracy:.4f}')
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return accuracy


def save_model(model, filename="decision_tree_model.pkl"):
    """Save the trained model to a file."""
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")
