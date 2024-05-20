import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import randint

preprocessed_data_path = "preprocessed_data.xlsx"
preprocessed_data = pd.read_excel(preprocessed_data_path)

X = preprocessed_data.iloc[:, 7:] 
y = preprocessed_data['oh_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)  
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

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
                                   scoring={'accuracy': 'accuracy',
                                            'precision': 'precision',
                                            'recall': 'recall',
                                            'f1_score': 'f1'},
                                   refit='accuracy', cv=5, verbose=2, random_state=42, n_jobs=-1)

random_search.fit(X_train_resampled, y_train_resampled)

print("Best parameters found by RandomizedSearchCV:")
print(random_search.best_params_)

best_random_forest = random_search.best_estimator_

best_random_forest.fit(X_train_resampled, y_train_resampled)

y_pred = best_random_forest.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Test Accuracy:", accuracy)
print("Test Precision:", precision)
print("Test Recall:", recall)
print("Test F1 Score:", f1)
