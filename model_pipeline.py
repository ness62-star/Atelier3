import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import signal
import time
from contextlib import contextmanager
from sklearn.preprocessing import LabelEncoder


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


def prepare_data(train_file, test_file):
    """
    Load and prepare data for training and testing.
    - Drops 'Area code'
    - Encodes 'International plan' and 'Voice mail plan' as binary
    - Replaces missing values with the mean
    - Drops rows with more than 50% missing values
    - Normalizes numerical features
    - Removes outliers (values more than 3 std devs from the mean)
    """
    # Load datasets
    print("Loading data...")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    # Combine datasets for consistent preprocessing
    combined_df = pd.concat([train_df, test_df])
    print(f"Combined dataset shape: {combined_df.shape}")
    # Drop 'Area code' column
    if 'Area code' in combined_df.columns:
        combined_df.drop(columns=['Area code'], inplace=True)
        print("Dropped 'Area code' column.")
    # Encode categorical features
    combined_df['International plan'] = combined_df['International plan'].map({'Yes': 1, 'No': 0})
    combined_df['Voice mail plan'] = combined_df['Voice mail plan'].map({'Yes': 1, 'No': 0})
    print("Encoded 'International plan' and 'Voice mail plan' as binary.")
    # Drop rows with more than 50% missing values
    thresh = len(combined_df.columns) // 2
    combined_df.dropna(thresh=thresh, inplace=True)
    print("Dropped rows with more than 50% missing values.")
    # Replace remaining NaNs with column mean
    combined_df.fillna(combined_df.mean(), inplace=True)
    print("Replaced remaining NaNs with column means.")
    # Separate features and target
    target = 'Churn'
    X = combined_df.drop(columns=[target])
    y = combined_df[target].map({'True': 1, 'False': 0})
    # Normalize numerical features
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    print("Normalized numerical features.")
    # Remove outliers (values > 3 standard deviations from mean)
    for col in numeric_cols:
        X = X[(X[col] < 3) & (X[col] > -3)]
    print("Removed outliers.")
    # Split back into training and testing sets
    X_train = X.iloc[:len(train_df)]
    y_train = y.iloc[:len(train_df)]
    X_test = X.iloc[len(train_df):]
    y_test = y.iloc[len(train_df):]
    print("Data preparation completed.")
    print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, 
                max_depth=5, 
                min_samples_split=20, 
                min_samples_leaf=10, 
                class_weight='balanced', 
                n_jobs=-1):
    """
    Train a Decision Tree with optimized parameters for speed and performance.
    - Limited tree depth
    - Minimum samples for split and leaf to avoid overfitting
    - Class weights to handle imbalance
    - Multi-processing for faster computation
    """
    print("Initializing Decision Tree Classifier...")
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=42
    )
    print("Starting model training...")
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Model training completed in {duration:.2f} seconds.")
    return model training completed in {duration:.2f} seconds.")
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
