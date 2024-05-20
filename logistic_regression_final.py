import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

preprocessed_data_path = 'preprocessed_data.xlsx'
preprocessed_data = pd.read_excel(preprocessed_data_path)

X = preprocessed_data.iloc[:, 7:]  
y = preprocessed_data['oh_label']

print("First column of X:", X.columns[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42) 
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

best_params = {'C': 3.594657285442726, 'penalty': 'l1', 'solver': 'liblinear'}

best_model = LogisticRegression(max_iter=2000, **best_params)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = cross_val_score(best_model, X_train_resampled, y_train_resampled, cv=kf, scoring='accuracy')
precision_scores = cross_val_score(best_model, X_train_resampled, y_train_resampled, cv=kf, scoring='precision')
recall_scores = cross_val_score(best_model, X_train_resampled, y_train_resampled, cv=kf, scoring='recall')

print("Average Validation Accuracy:", accuracy_scores.mean())
print("Average Validation Precision:", precision_scores.mean())
print("Average Validation Recall:", recall_scores.mean())

best_model.fit(X_train_resampled, y_train_resampled)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy on test set:", accuracy)
print("Precision on test set:", precision)
print("Recall on test set:", recall)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
