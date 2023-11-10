from Utils import *

modal = ['sequence', 'graph', 'point_cloud', 'multimodal']
modal_id = 3

data_folder = 'downstreamtasks/ProtDD/'
with open(f'{data_folder}{modal[modal_id]}.pkl', 'rb') as f:
    X = np.array(pickle.load(f))

df = pd.read_csv(f'{data_folder}/label.csv')
y = df['label']
fold_ids = df['fold_id']
cv = GroupKFold(n_splits=10)

def train():
    for i, (train_index, test_index) in enumerate(cv.split(X, y, groups=fold_ids)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, random_state=42, tree_method='gpu_hist', objective="binary:logistic")
        model.fit(X_train, y_train, eval_metric='logloss', eval_set= [(X_test, y_test)], early_stopping_rounds=10, verbose = 0)
        model.save_model(f"{data_folder}{modal[modal_id]}_model/xgb_model_fold{i+1}.json")

def test():
    fold_scores = []
    for i, (train_index, test_index) in enumerate(cv.split(X, y, groups=fold_ids)):
        X_test, y_test  = X[test_index], y.iloc[test_index] 
        model = XGBClassifier()
        model.load_model(f"{data_folder}{modal[modal_id]}_model/xgb_model_fold{i+1}.json")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Fold {i+1} Accuracy: {accuracy:.4f}")
        fold_scores .append(accuracy)
    print("Mean Accuracy:", np.mean(fold_scores))
