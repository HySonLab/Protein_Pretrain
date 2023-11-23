from Utils import *
data_path = 'data/Atom3D_MSP/split-by-sequence-identity-30/data/'
X = {}
y = {}
folders = ['train', 'val', 'test']
modal = ['sequence', 'graph', 'point_cloud', 'multimodal']
modal_id = 2


for folder in folders:
    with open(f'{data_path}{folder}/{modal[modal_id]}.pkl', 'rb') as f:
        X[folder] = np.array(pickle.load(f))
   
    with open(f'{data_path}{folder}/label.pkl', 'rb') as f:
        y[folder] = np.array(pickle.load(f))


X_train, y_train = X['train'], y['train']
X_val, y_val = X['val'], y['val']
X_test, y_test = X['test'], y['test']
print("Data loaded successfully")


xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=10, random_state=42, tree_method='gpu_hist', objective='binary:logistic')
xgb_model.fit(X_train, y_train, eval_metric='aucpr', eval_set= [(X_train, y_train), (X_val, y_val)], early_stopping_rounds=10, verbose = 1)


xgb_model.save_model(f"{data_path}{modal[modal_id]}_model.json")
xgb_model = XGBClassifier()
xgb_model.load_model(f"{data_path}{modal[modal_id]}_model.json")


y_pred = xgb_model.predict(X_test)
print("AUROC on test set:", roc_auc_score(y_test, y_pred))



