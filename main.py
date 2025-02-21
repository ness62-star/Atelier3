from model_pipeline import prepare_data, train_model, evaluate_model, save_model
import argparse


def main():
    """Main function to train and evaluate the Decision Tree model."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Decision Tree Churn Prediction")
    parser.add_argument("--train_file", type=str, default="churn-bigml-80.csv",
                        help="Path to training dataset")
    parser.add_argument("--test_file", type=str, default="churn-bigml-20.csv",
                        help="Path to testing dataset")
    parser.add_argument("--max_depth", type=int, default=3,
                        help="Maximum depth of the decision tree")
    parser.add_argument("--save", action='store_true',
                        help="Flag to save the trained model")
    parser.add_argument("--prepare_data", action='store_true',
                        help="Flag to prepare data only")
    parser.add_argument("--sample_fraction", type=float, default=1.0,
                        help="Fraction of data to use for training")
    args = parser.parse_args()
    # Prepare data with specified sample fraction
    X_train, X_test, y_train, y_test = prepare_data(
        args.train_file, 
        args.test_file, 
        sample_fraction=args.sample_fraction
    )
    print(f"Training on {len(X_train)} samples...")    
    if args.prepare_data:
        print("Data preparation completed successfully.")
        return
    # Train model with timeout
    try:
        model = train_model(
            X_train, 
            y_train, 
            max_depth=args.max_depth,
            timeout=300  # 5 minute timeout
        )
        # Evaluate model
        accuracy = evaluate_model(model, X_test, y_test)
        print(f"Model accuracy: {accuracy:.2f}")
        # Save model if specified
        if args.save:
            save_model(model) 
    except TimeoutError as e:
        print(f"Training error: {e}")
        return
if __name__ == "__main__":
    main()
