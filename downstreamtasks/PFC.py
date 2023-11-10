from Utils import *

data_folder = '/downstreamtasks/data/SCOPe1.75/'
label_file_name = ['training', 'validation', 'test_family', 'test_fold', 'test_superfamily']
modal = ['sequence', 'graph', 'point_cloud', 'multimodal']
modal_id = 0

X = {}
y = {}
for file in label_file_name:
    with open(f'{data_folder}/{file}.txt', 'r') as f:
        lines = f.readlines()
        y[file] = [line.split()[-1] for line in lines]

    with open(f'{data_folder}{file}/{modal[modal_id]}.pkl', 'rb') as f:
        X[file] = pickle.load(f)


X_train, y_train = X["training"], y['training']
X_validation, y_validation = X["validation"], y['validation']

X_test1, y_test1 = X["test_family"], y['test_family']
X_test2, y_test2 = X["test_fold"], y['test_fold']
X_test3, y_test3 = X["test_superfamily"], y['test_superfamily']

label = set(y_train)
label_encoder = LabelEncoder()
label_encoder.fit(list(label))

y_train = label_encoder.transform(y_train)
y_validation = label_encoder.transform(y_validation)
y_test1 = label_encoder.transform(y_test1)
y_test2 = label_encoder.transform(y_test2)
y_test3 = label_encoder.transform(y_test3)

def train():
    xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, random_state=42, tree_method='gpu_hist', objective='multi:softmax')
    xgb_model.fit(X_train, y_train, eval_metric='mlogloss', eval_set= [(X_validation, y_validation)], early_stopping_rounds=10, verbose = 1)
    xgb_model.save_model(f"{data_folder}{modal[modal_id]}_model.json")

def test():
    xgb_model = XGBClassifier()
    xgb_model.load_model(f"{data_folder}/{modal[modal_id]}_model.json")

    y_pred = xgb_model.predict(X_test1)
    print("Accuracy on test_family:", accuracy_score(y_test1, y_pred))

    y_pred = xgb_model.predict(X_test2)
    print("Accuracy on test_fold:", accuracy_score(y_test2, y_pred))

    y_pred = xgb_model.predict(X_test3)
    print("Accuracy on test_superfamily:", accuracy_score(y_test3, y_pred))