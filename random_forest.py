import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import randint
from imblearn.under_sampling import EditedNearestNeighbours
import matplotlib.pyplot as plt

preprocessed_data_path_word2vec = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_train_data_word2vec.csv"
test_data_word2vec_path = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_test_data_word2vec.csv"

preprocessed_data_word2vec = pd.read_csv(preprocessed_data_path_word2vec)
test_data_word2vec = pd.read_csv(test_data_word2vec_path)

X_word2vec_with_sentiment = preprocessed_data_word2vec.iloc[:, :-1]
y_word2vec = preprocessed_data_word2vec.iloc[:, -1]

X_test_word2vec_with_sentiment = test_data_word2vec.iloc[:, :-1]
y_test_word2vec = test_data_word2vec.iloc[:, -1]

def run_model_and_evaluate(X, y, X_test, y_test, method_name):
    random_forest = RandomForestClassifier(random_state=42)
    param_dist = {
        'n_estimators': randint(50, 500),
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None] + list(range(10, 110, 10)),
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    random_search = RandomizedSearchCV(estimator=random_forest, param_distributions=param_dist, n_iter=100,
                                       scoring={'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall'},
                                       refit='accuracy', cv=5, verbose=2, random_state=42, n_jobs=-1)
    random_search.fit(X, y)
    best_model = random_search.best_estimator_
    cv_accuracy = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
    cv_precision = cross_val_score(best_model, X, y, cv=5, scoring='precision')
    cv_recall = cross_val_score(best_model, X, y, cv=5, scoring='recall')
    best_model.fit(X, y)
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    print(f"Results for {method_name}:")
    print("Best Hyperparameters:", random_search.best_params_)
    print("Cross-validation Accuracy:", cv_accuracy.mean())
    print("Cross-validation Precision:", cv_precision.mean())
    print("Cross-validation Recall:", cv_recall.mean())
    print("Test set Accuracy:", test_accuracy)
    print("Test set Precision:", test_precision)
    print("Test set Recall:", test_recall)
    print()

enn = EditedNearestNeighbours()
X_resampled_with_sentiment, y_resampled_with_sentiment = enn.fit_resample(X_word2vec_with_sentiment, y_word2vec)
run_model_and_evaluate(X_resampled_with_sentiment, y_resampled_with_sentiment, X_test_word2vec_with_sentiment, y_test_word2vec, "ENN with Sentiment Analysis")
