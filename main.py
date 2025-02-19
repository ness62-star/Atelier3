from model_pipeline import prepare_data, train_model, evaluate_model, save_model
import argparse


def main():
    """Main function to train and evaluate the Decision Tree model."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Decision Tree Churn Prediction")
    parser.add_argument("--train_file", type=str, default="churn-bigml-80.csv", help="Path to training dataset")
    parser.add_argument("--test_file", type=str, default="churn-bigml-20.csv", help="Path to testing dataset")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of the decision tree")
    parser.add_argument("--save", action='store_true', help="Flag to save the trained model")
    args = parser.parse_args()
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(args.train_file, args.test_file)
    # Train model
    model = train_model(X_train, y_train, max_depth=args.max_depth)
    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    # Print model accuracy
    print(f"Précision du modèle : {accuracy:.2f}")
    # Save model if specified
    if args.save:
        save_model(model)


if __name__ == "__main__":
    main()
