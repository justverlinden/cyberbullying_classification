import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import uniform

preprocessed_data_path = 'preprocessed_data.xlsx'
preprocessed_data = pd.read_excel(preprocessed_data_path)

X = preprocessed_data.iloc[:, 7:]
y = preprocessed_data['oh_label']

print("First column of X:", X.columns[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

param_distributions = [
    {'penalty': ['l2'], 'solver': ['lbfgs'], 'C': uniform(0.01, 10)},
    {'penalty': ['l1', 'l2'], 'solver': ['liblinear'], 'C': uniform(0.01, 10)},
    {'penalty': ['l1', 'l2'], 'solver': ['saga'], 'C': uniform(0.01, 10)},
    {'penalty': ['elasticnet'], 'solver': ['saga'], 'C': uniform(0.01, 10), 'l1_ratio': uniform(0, 1)}
]

logistic_model = LogisticRegression(max_iter=2000)

random_search = RandomizedSearchCV(logistic_model, param_distributions, n_iter=50, cv=5, random_state=42, n_jobs=-1)
random_search.fit(X_train_resampled, y_train_resampled)

print("Best parameters found:", random_search.best_params_)

best_model = random_search.best_estimator_
best_model.fit(X_train_resampled, y_train_resampled)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy on test set:", accuracy)
print("Precision on test set:", precision)
print("Recall on test set:", recall)
