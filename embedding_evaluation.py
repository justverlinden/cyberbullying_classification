import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# Load the preprocessed data
preprocessed_data_path_word2vec = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_train_data_word2vec.csv"
preprocessed_data_path_tfidf = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_train_data_tfidf.csv"
preprocessed_data_path_bert = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_train_data_bert.csv"
train_tokenized_path = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_train_data_tokenized.csv"

test_data_word2vec_path = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_test_data_word2vec.csv"
test_data_tfidf_path = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_test_data_tfidf.csv"
test_data_bert_path = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_test_data_bert.csv"
test_tokenized_path = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_test_data_tokenized.csv"

preprocessed_data_word2vec = pd.read_csv(preprocessed_data_path_word2vec)
preprocessed_data_tfidf = pd.read_csv(preprocessed_data_path_tfidf)
preprocessed_data_bert = pd.read_csv(preprocessed_data_path_bert)
train_data_tokenized = pd.read_csv(train_tokenized_path)

test_data_word2vec = pd.read_csv(test_data_word2vec_path)
test_data_tfidf = pd.read_csv(test_data_tfidf_path)
test_data_bert = pd.read_csv(test_data_bert_path)
test_data_tokenized = pd.read_csv(test_tokenized_path)

# Separate X and y
X_word2vec = preprocessed_data_word2vec.iloc[:, :-1]
y_word2vec = preprocessed_data_word2vec.iloc[:, -1]

X_tfidf = preprocessed_data_tfidf.iloc[:, :-1]
y_tfidf = preprocessed_data_tfidf.iloc[:, -1]

X_bert = preprocessed_data_bert.iloc[:, :-1]
y_bert = preprocessed_data_bert.iloc[:, -1]

X_train_tokenized = train_data_tokenized.drop(columns=['oh_label'])
y_train_tokenized = train_data_tokenized['oh_label']

X_test_word2vec = test_data_word2vec.iloc[:, :-1]
y_test_word2vec = test_data_word2vec.iloc[:, -1]

X_test_tfidf = test_data_tfidf.iloc[:, :-1]
y_test_tfidf = test_data_tfidf.iloc[:, -1]

X_test_bert = test_data_bert.iloc[:, :-1]
y_test_bert = test_data_bert.iloc[:, -1]

X_test_tokenized = test_data_tokenized.drop(columns=['oh_label'])
y_test_tokenized = test_data_tokenized['oh_label']

# Feature scaling for tokenized data
scaler = StandardScaler()
X_train_tokenized_scaled = scaler.fit_transform(X_train_tokenized)
X_test_tokenized_scaled = scaler.transform(X_test_tokenized)

# Define function to run logistic regression and evaluate
def run_logistic_regression(X, y, X_test, y_test):
    logistic_model = LogisticRegression(max_iter=2000)
    cv_accuracy = cross_val_score(logistic_model, X, y, cv=5, scoring='accuracy')
    cv_precision = cross_val_score(logistic_model, X, y, cv=5, scoring='precision')
    cv_recall = cross_val_score(logistic_model, X, y, cv=5, scoring='recall')
    logistic_model.fit(X, y)
    y_pred = logistic_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    return cv_accuracy.mean(), cv_precision.mean(), cv_recall.mean(), test_accuracy, test_precision, test_recall

# Define function to run random forest and evaluate
def run_random_forest(X, y, X_test, y_test):
    rf_model = RandomForestClassifier()
    cv_accuracy = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
    cv_precision = cross_val_score(rf_model, X, y, cv=5, scoring='precision')
    cv_recall = cross_val_score(rf_model, X, y, cv=5, scoring='recall')
    rf_model.fit(X, y)
    y_pred = rf_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    return cv_accuracy.mean(), cv_precision.mean(), cv_recall.mean(), test_accuracy, test_precision, test_recall

# Run models on all embeddings
embeddings = {
    "Word2Vec": (X_word2vec, y_word2vec, X_test_word2vec, y_test_word2vec),
    "TF-IDF": (X_tfidf, y_tfidf, X_test_tfidf, y_test_tfidf),
    "BERT": (X_bert, y_bert, X_test_bert, y_test_bert),
    "Tokenized + Padded": (X_train_tokenized_scaled, y_train_tokenized, X_test_tokenized_scaled, y_test_tokenized)
}

for name, (X_train, y_train, X_test, y_test) in embeddings.items():
    lr_results = run_logistic_regression(X_train, y_train, X_test, y_test)
    rf_results = run_random_forest(X_train, y_train, X_test, y_test)
    
    print(f"Logistic Regression with {name} embeddings:")
    print(f"  Cross-validation Accuracy: {lr_results[0]:.4f}")
    print(f"  Cross-validation Precision: {lr_results[1]:.4f}")
    print(f"  Cross-validation Recall: {lr_results[2]:.4f}")
    print(f"  Test set Accuracy: {lr_results[3]:.4f}")
    print(f"  Test set Precision: {lr_results[4]:.4f}")
    print(f"  Test set Recall: {lr_results[5]:.4f}")
    print()
    
    print(f"Random Forest with {name} embeddings:")
    print(f"  Cross-validation Accuracy: {rf_results[0]:.4f}")
    print(f"  Cross-validation Precision: {rf_results[1]:.4f}")
    print(f"  Cross-validation Recall: {rf_results[2]:.4f}")
    print(f"  Test set Accuracy: {rf_results[3]:.4f}")
    print(f"  Test set Precision: {rf_results[4]:.4f}")
    print(f"  Test set Recall: {rf_results[5]:.4f}")
    print()
