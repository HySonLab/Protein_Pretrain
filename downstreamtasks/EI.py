from Utils import *

# Specify the modalities and the selected modality index
modal = ['sequence', 'graph', 'point_cloud', 'multimodal']
modal_id = 3

data_folder = '/data/ProtDD'

# Load feature data from the selected modality
with open(f'{data_folder}/{modal[modal_id]}.pkl', 'rb') as f:
    X = np.array(pickle.load(f))

# Read the label CSV file
df = pd.read_csv(f'{data_folder}/label.csv')
print("Number of samples:", len(df))
y = df['label']
fold_ids = df['fold_id']
cv = GroupKFold(n_splits=10)

def train():
    # Train a model for each fold
    for i, (train_index, test_index) in enumerate(cv.split(X, y, groups=fold_ids)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Define and train the XGBoost classifier
        model = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, random_state=42, tree_method='gpu_hist', objective="binary:logistic")
        model.fit(X_train, y_train, eval_metric='logloss', eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=10, verbose=0)
        
        # Save the trained model for this fold
        model.save_model(f"{data_folder}{modal[modal_id]}_model/xgb_model_fold{i+1}.json")

def test():
    fold_scores = []
    for i, (train_index, test_index) in enumerate(cv.split(X, y, groups=fold_ids)):
        X_test, y_test = X[test_index], y.iloc[test_index]
        
        # Load the trained XGBoost classifier model
        model = XGBClassifier()
        model.load_model(f"{data_folder}{modal[modal_id]}_model/xgb_model_fold{i+1}.json")
        
        # Make predictions and calculate accuracy for each fold
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Fold {i+1} Accuracy: {accuracy:.4f}")
        fold_scores.append(accuracy)
    
    # Calculate and print the mean accuracy over all folds
    print("Mean Accuracy:", np.mean(fold_scores))
