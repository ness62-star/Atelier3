import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def prepare_data(train_file, test_file):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Séparation des features et de la cible
    target = 'Churn'  # Mets ici le nom exact de la colonne cible
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # Encodage des colonnes catégoriques
    label_encoders = {}
    for col in X_train.columns:
        if X_train[col].dtype == 'object':  # Vérifie si la colonne est catégorique
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])
            X_test[col] = le.transform(X_test[col])
            label_encoders[col] = le  # Stocke l'encodeur pour réutilisation

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, max_depth=5, min_samples_split=20, min_samples_leaf=10):
    """
    Train a Decision Tree with controlled depth and minimum samples 
    to prevent infinite loops and overfitting.
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,          # Limit the depth to prevent overfitting
        min_samples_split=min_samples_split,  # Minimum samples required to split
        min_samples_leaf=min_samples_leaf,    # Minimum samples at leaf node
        random_state=42
    )
    model.fit(X_train, y_train)
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
