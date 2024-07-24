from Utils import *

def load_data(modal, folders):
    # Define the path where data is stored
    data_path = './downstreamtasks/data/Atom3D_MSP/split-by-sequence-identity-30/data'
    
    # Initialize dictionaries to store data and labels for different folders
    X = {}
    y = {}

    # Load data and labels for each folder
    for folder in folders:
        with open(f'{data_path}/{folder}/{modal}.pkl', 'rb') as f:
            X[folder] = np.array(pickle.load(f))

        with open(f'{data_path}/{folder}/label.pkl', 'rb') as f:
            y[folder] = np.array(pickle.load(f))

    return X, y

def train(data_path, modal, X_train, y_train, X_val, y_val):
    # Initialize XGBoost classifier with specified hyperparameters
    xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=10, random_state=42, tree_method='gpu_hist', objective='binary:logistic')

    # Train the model using training data and validate on a separate validation set
    xgb_model.fit(X_train, y_train, eval_metric='aucpr', eval_set=[(X_train, y_train), (X_val, y_val)], early_stopping_rounds=10, verbose=1)

    # Save the trained model in JSON format
    xgb_model.save_model(f"{data_path}/{modal}_model.json")

def test(data_path, modal, X_test, y_test):
    # Initialize XGBoost classifier
    xgb_model = XGBClassifier()

    # Load a pre-trained model from the specified path
    xgb_model.load_model(f"{data_path}/{modal}_model.json")

    # Make predictions on the test set
    y_pred = xgb_model.predict(X_test)

    # Print the AUROC score on the test set
    print("AUROC on test set:", roc_auc_score(y_test, y_pred))

def main():
    # Parse command line arguments for modality and mode (train or test)
    parser = argparse.ArgumentParser(description='Train and test XGBoost models with different modalities.')
    parser.add_argument('--modal', type=str, choices=['sequence', 'graph', 'point_cloud', 'multimodal'], default='multimodal', help='Select the modality for training and testing.')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='Select the mode (train or test).')
    args = parser.parse_args()

    # Load data and labels for train, validation, and test sets
    X, y = load_data(args.modal, ['train', 'val', 'test'])

    if args.mode == 'train':
        # Train the XGBoost model
        train(args.data_path, args.modal, X['train'], y['train'], X['val'], y['val'])
    elif args.mode == 'test':
        # Test the XGBoost model
        test(args.data_path, args.modal, X['test'], y['test'])
    else:
        # Handle invalid mode
        print("Invalid mode. Use 'train' or 'test'.")

if __name__ == "__main__":
    main()
