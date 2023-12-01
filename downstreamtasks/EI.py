from Utils import *

def load_data(modal, data_folder):
    # Load input data (X) from a pickle file and labels (y) from a CSV file
    with open(f'{data_folder}/{modal}.pkl', 'rb') as f:
        X = np.array(pickle.load(f))

    df = pd.read_csv(f'{data_folder}/label.csv')
    y = df['label']
    fold_ids = df['fold_id']
    
    return X, y, fold_ids

def train(modal, data_folder):
    # Load data and perform 10-fold cross-validation
    X, y, fold_ids = load_data(modal, data_folder)
    cv = GroupKFold(n_splits=10)

    for i, (train_index, test_index) in enumerate(cv.split(X, y, groups=fold_ids)):
        # Split the data into training and testing sets for each fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Initialize and train the XGBoost classifier
        model = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, random_state=42, tree_method='gpu_hist', objective="binary:logistic")
        model.fit(X_train, y_train, eval_metric='logloss', eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=10, verbose=0)

        # Save the trained model for each fold
        model.save_model(f"{data_folder}{modal}_model/xgb_model_fold{i+1}.json")

def test(modal, data_folder):
    # Load data and perform 10-fold cross-validation for testing
    X, y, fold_ids = load_data(modal, data_folder)
    cv = GroupKFold(n_splits=10)

    fold_scores = []
    for i, (train_index, test_index) in enumerate(cv.split(X, y, groups=fold_ids)):
        # Load the trained model for each fold
        X_test, y_test = X[test_index], y.iloc[test_index]
        model = XGBClassifier()
        model.load_model(f"{data_folder}{modal}_model/xgb_model_fold{i+1}.json")

        # Make predictions and calculate accuracy for each fold
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Fold {i+1} Accuracy: {accuracy:.4f}")
        fold_scores.append(accuracy)

    # Print the mean accuracy across all folds
    print("Mean Accuracy:", np.mean(fold_scores))

def main():
    # Parse command line arguments for modality and mode (train or test)
    parser = argparse.ArgumentParser(description='Train and test models with different modalities.')
    parser.add_argument('--modal', type=str, choices=['sequence', 'graph', 'point_cloud', 'multimodal'], default='multimodal', help='Select the modality for training and testing.')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='Select the mode (train or test).')
    args = parser.parse_args()

    data_folder = '/data/ProtDD'
    modal = args.modality

    if args.mode == 'train':
        # Train the XGBoost model
        train(modal, data_folder)
    elif args.mode == 'test':
        # Test the XGBoost model
        test(modal, data_folder)
    else:
        # Handle invalid mode
        print("Invalid mode. Use 'train' or 'test'.")

if __name__ == "__main__":
    main()
