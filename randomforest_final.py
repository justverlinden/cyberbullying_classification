import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

preprocessed_data_path = "preprocessed_data.xlsx"
preprocessed_data = pd.read_excel(preprocessed_data_path)

X = preprocessed_data.iloc[:, 7:]  
y = preprocessed_data['oh_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)  
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

best_params = {
    'bootstrap': False,
    'max_depth': 30,
    'max_features': 'sqrt',
    'min_samples_leaf': 2,
    'min_samples_split': 5,
    'n_estimators': 481
}

best_random_forest = RandomForestClassifier(**best_params, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {
    'accuracy': [],
    'precision': [],
    'recall': []
}

for train_index, val_index in kf.split(X_train_resampled):
    X_train_fold, X_val_fold = X_train_resampled.iloc[train_index], X_train_resampled.iloc[val_index]
    y_train_fold, y_val_fold = y_train_resampled.iloc[train_index], y_train_resampled.iloc[val_index]
    
    best_random_forest.fit(X_train_fold, y_train_fold)
    
    y_val_pred = best_random_forest.predict(X_val_fold)
    
    cv_results['accuracy'].append(accuracy_score(y_val_fold, y_val_pred))
    cv_results['precision'].append(precision_score(y_val_fold, y_val_pred))
    cv_results['recall'].append(recall_score(y_val_fold, y_val_pred))

average_cv_results = {metric: sum(scores) / len(scores) for metric, scores in cv_results.items()}

print("Average Cross-Validation Accuracy:", average_cv_results['accuracy'])
print("Average Cross-Validation Precision:", average_cv_results['precision'])
print("Average Cross-Validation Recall:", average_cv_results['recall'])

best_random_forest.fit(X_train_resampled, y_train_resampled)

y_test_pred = best_random_forest.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)

print("Test Accuracy:", test_accuracy)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)

cm = confusion_matrix(y_test, y_test_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
