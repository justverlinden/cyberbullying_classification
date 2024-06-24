import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.under_sampling import EditedNearestNeighbours

# Update file paths based on your local machine
preprocessed_data_path_word2vec = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_train_data_word2vec.csv"
test_data_word2vec_path = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_test_data_word2vec.csv"

# Load preprocessed data
preprocessed_data_word2vec = pd.read_csv(preprocessed_data_path_word2vec)
test_data_word2vec = pd.read_csv(test_data_word2vec_path)

# Separate X and y
X_word2vec_with_sentiment = preprocessed_data_word2vec.iloc[:, :-1]  # All columns except the last one (target variable)
y_word2vec = preprocessed_data_word2vec.iloc[:, -1]   # Last column is the target variable

X_test_word2vec_with_sentiment = test_data_word2vec.iloc[:, :-1]  # All columns except the last one (target variable)
y_test_word2vec = test_data_word2vec.iloc[:, -1]   # Last column is the target variable

# Drop the new last column (sentiment analysis column) to create datasets without sentiment analysis
X_word2vec_without_sentiment = X_word2vec_with_sentiment.drop(X_word2vec_with_sentiment.columns[-1], axis=1)
X_test_word2vec_without_sentiment = X_test_word2vec_with_sentiment.drop(X_test_word2vec_with_sentiment.columns[-1], axis=1)

# Define function to run logistic regression and evaluate
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

# Define function to run random forest and evaluate
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

# Apply ENN undersampling and evaluate with and without sentiment analysis
enn = EditedNearestNeighbours()

# With sentiment analysis
X_resampled_with_sentiment, y_resampled_with_sentiment = enn.fit_resample(X_word2vec_with_sentiment, y_word2vec)
run_logistic_regression(X_resampled_with_sentiment, y_resampled_with_sentiment, X_test_word2vec_with_sentiment, y_test_word2vec, "ENN with Sentiment Analysis")
run_random_forest(X_resampled_with_sentiment, y_resampled_with_sentiment, X_test_word2vec_with_sentiment, y_test_word2vec, "ENN with Sentiment Analysis")

# Without sentiment analysis
X_resampled_without_sentiment, y_resampled_without_sentiment = enn.fit_resample(X_word2vec_without_sentiment, y_word2vec)
run_logistic_regression(X_resampled_without_sentiment, y_resampled_without_sentiment, X_test_word2vec_without_sentiment, y_test_word2vec, "ENN without Sentiment Analysis")
run_random_forest(X_resampled_without_sentiment, y_resampled_without_sentiment, X_test_word2vec_without_sentiment, y_test_word2vec, "ENN without Sentiment Analysis")
