from Utils import *

# Specify the data folder and the label file names
data_folder = '/data/SCOPe1.75/'
label_file_name = ['training', 'validation', 'test_family', 'test_fold', 'test_superfamily']

# Specify the modality and its ID
modal = ['sequence', 'graph', 'point_cloud', 'multimodal']
modal_id = 0

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
    with open(f'{data_folder}{file}/{modal[modal_id]}.pkl', 'rb') as f:
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

def train():
    # Train a multi-class classification model
    xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, random_state=42, tree_method='gpu_hist', objective='multi:softmax')
    xgb_model.fit(X_train, y_train, eval_metric='mlogloss', eval_set=[(X_validation, y_validation)], early_stopping_rounds=10, verbose=1)
    
    # Save the trained model
    xgb_model.save_model(f"{data_folder}{modal[modal_id]}_model.json")

def test():
    # Load the trained XGBoost classifier model
    xgb_model = XGBClassifier()
    xgb_model.load_model(f"{data_folder}/{modal[modal_id]}_model.json")

    # Make predictions and calculate accuracy for each test set
    y_pred = xgb_model.predict(X_test1)
    print("Accuracy on test_family:", accuracy_score(y_test1, y_pred))

    y_pred = xgb_model.predict(X_test2)
    print("Accuracy on test_fold:", accuracy_score(y_test2, y_pred))

    y_pred = xgb_model.predict(X_test3)
    print("Accuracy on test_superfamily:", accuracy_score(y_test3, y_pred))
