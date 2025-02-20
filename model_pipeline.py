import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import signal
import time
from contextlib import contextmanager


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


def prepare_data(train_file, test_file, sample_fraction=0.1):
    """
    Load and prepare data for training and testing.
    Use a subset of data for debugging long training times.
    """
    print("Loading training data...")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    # Debug: Use only 10% of the data
    print(f"Using {sample_fraction*100}% of the data for debugging...")
    train_df = train_df.sample(frac=sample_fraction, random_state=42)
    test_df = test_df.sample(frac=sample_fraction, random_state=42)
    # Separate features and target
    target = 'Churn'
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    print("Data preparation completed.")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, max_depth=5, min_samples_split=20, min_samples_leaf=10):
    """
    Train a Decision Tree with controlled depth and minimum samples.
    Added detailed logging to track progress.
    """
    print("Initializing Decision Tree Classifier...")
    start_time = time.time()
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    print("Starting model training...")
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
