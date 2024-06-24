import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours

preprocessed_data_path_word2vec = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_train_data_word2vec.csv"
test_data_word2vec_path = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_test_data_word2vec.csv"

preprocessed_data_word2vec = pd.read_csv(preprocessed_data_path_word2vec)
test_data_word2vec = pd.read_csv(test_data_word2vec_path)

X_word2vec = preprocessed_data_word2vec.iloc[:, :-1]
y_word2vec = preprocessed_data_word2vec.iloc[:, -1]

X_test_word2vec = test_data_word2vec.iloc[:, :-1]
y_test_word2vec = test_data_word2vec.iloc[:, -1]

def run_logistic_regression(X, y, X_test, y_test, method_name):
    logistic_model = LogisticRegression(max_iter=2000, random_state=42)
    cv_accuracy = cross_val_score(logistic_model, X, y, cv=5, scoring='accuracy')
    cv_precision = cross_val_score(logistic_model, X, y, cv=5, scoring='precision')
    cv_recall = cross_val_score(logistic_model, X, y, cv=5, scoring='recall')
    logistic_model.fit(X, y)
    y_pred = logistic_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    print(f"Logistic Regression Results for {method_name}:")
    print("Cross-validation Accuracy:", cv_accuracy.mean())
    print("Cross-validation Precision:", cv_precision.mean())
    print("Cross-validation Recall:", cv_recall.mean())
    print("Test set Accuracy:", test_accuracy)
    print("Test set Precision:", test_precision)
    print("Test set Recall:", test_recall)
    print()

def run_random_forest(X, y, X_test, y_test, method_name):
    rf_model = RandomForestClassifier(random_state=42)
    cv_accuracy = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
    cv_precision = cross_val_score(rf_model, X, y, cv=5, scoring='precision')
    cv_recall = cross_val_score(rf_model, X, y, cv=5, scoring='recall')
    rf_model.fit(X, y)
    y_pred = rf_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    print(f"Random Forest Results for {method_name}:")
    print("Cross-validation Accuracy:", cv_accuracy.mean())
    print("Cross-validation Precision:", cv_precision.mean())
    print("Cross-validation Recall:", cv_recall.mean())
    print("Test set Accuracy:", test_accuracy)
    print("Test set Precision:", test_precision)
    print("Test set Recall:", test_recall)
    print()

undersampling_methods = [
    (RandomUnderSampler(sampling_strategy=0.5, random_state=42), "Random Undersampling"),
    (NearMiss(version=1, sampling_strategy=0.5), "NearMiss"),
    (TomekLinks(), "Tomek Links"),
    (EditedNearestNeighbours(), "Edited Nearest Neighbor (ENN)")
]

for sampler, method_name in undersampling_methods:
    X_resampled, y_resampled = sampler.fit_resample(X_word2vec, y_word2vec)
    run_logistic_regression(X_resampled, y_resampled, X_test_word2vec, y_test_word2vec, method_name)
    run_random_forest(X_resampled, y_resampled, X_test_word2vec, y_test_word2vec, method_name)
