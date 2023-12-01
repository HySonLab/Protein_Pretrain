from Utils import *

def train(data_folder, modal, X_train, y_train, X_validation, y_validation):
    # Train a multi-class classification model using XGBoost
    xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, random_state=42, tree_method='gpu_hist', objective='multi:softmax')
    xgb_model.fit(X_train, y_train, eval_metric='mlogloss', eval_set=[(X_train, y_train), (X_validation, y_validation)], early_stopping_rounds=10, verbose=1)
    
    # Save the trained model
    xgb_model.save_model(f"{data_folder}/{modal}_model.json")

def test(data_folder, modal, X_test, y_test):
    # Load the trained XGBoost classifier model
    xgb_model = XGBClassifier()
    xgb_model.load_model(f"{data_folder}/{modal}_model.json")

    # Make predictions and calculate accuracy for each test set
    y_pred = xgb_model.predict(X_test)
    print(f"Accuracy :", accuracy_score(y_test, y_pred))

def main():
    # Parse command line arguments for modality, mode (train or test), and test dataset
    parser = argparse.ArgumentParser(description='Train and test XGBoost models with different modalities.')
    parser.add_argument('--modal', type=str, choices=['sequence', 'graph', 'point_cloud', 'multimodal'], default='multimodal', help='Select the modality for training and testing.')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='Select the mode (train or test).')
    parser.add_argument('--test_dataset', type=str, choices=['test_family', 'test_fold', 'test_superfamily'], default='test_family', help='Select the test dataset for testing.')
    args = parser.parse_args()

    # Specify the data folder and the label file names
    data_folder = '/data/SCOPe1.75'
    label_file_name = ['training', 'validation', 'test_family', 'test_fold', 'test_superfamily']

    # Initialize dictionaries to store feature data and labels
    X = {}
    y = {}

    # Load feature data and labels for each label file
    for file in label_file_name:
        # Load labels from the text file
        with open(f'{data_folder}/{file}.txt', 'r') as f:
            lines = f.readlines()
            y[file] = [line.split()[-1] for line in lines]

        # Load feature data from the selected modality
        with open(f'{data_folder}{file}/{args.modal}.pkl', 'rb') as f:
            X[file] = pickle.load(f)

    # Split data into train, validation, and test sets
    X_train, y_train = X["training"], y['training']
    X_validation, y_validation = X["validation"], y['validation']

    X_test1, y_test1 = X["test_family"], y['test_family']
    X_test2, y_test2 = X["test_fold"], y['test_fold']
    X_test3, y_test3 = X["test_superfamily"], y['test_superfamily']

    # Encode labels using LabelEncoder
    label = set(y_train)
    label_encoder = LabelEncoder()
    label_encoder.fit(list(label))

    y_train = label_encoder.transform(y_train)
    y_validation = label_encoder.transform(y_validation)
    y_test1 = label_encoder.transform(y_test1)
    y_test2 = label_encoder.transform(y_test2)
    y_test3 = label_encoder.transform(y_test3)

    if args.mode == 'train':
        # Train the XGBoost model
        train(data_folder, args.modal, X_train, y_train, X_validation, y_validation)
    elif args.mode == 'test':
        # Test the XGBoost model on the selected test dataset
        if args.test_dataset == 'test_family':
            test(data_folder, args.modal, X_test1, y_test1)
        elif args.test_dataset == 'test_fold':
            test(data_folder, args.modal, X_test2, y_test2)
        elif args.test_dataset == 'test_superfamily':
            test(data_folder, args.modal, X_test3, y_test3)
        else:
            print("Invalid test dataset. Use 'test_family', 'test_fold', or 'test_superfamily'.")
    else:
        # Handle invalid mode
        print("Invalid mode. Use 'train' or 'test'.")

if __name__ == "__main__":
    main()
