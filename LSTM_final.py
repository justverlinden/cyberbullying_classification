import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

preprocessed_data_path = "cleaned_data.xlsx"
preprocessed_data = pd.read_excel(preprocessed_data_path)

preprocessed_data.dropna(subset=['filtered_text'], inplace=True)

X = preprocessed_data['filtered_text']
y = preprocessed_data['oh_label']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)

max_sequence_length = max([len(seq) for seq in X_sequences])
X_padded = pad_sequences(X_sequences, maxlen=max_sequence_length)

y = y.reset_index(drop=True)

X_train_full, X_test, y_train_full, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

def build_best_model():
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32, input_length=max_sequence_length))
    
    for _ in range(2):  
        model.add(LSTM(units=224, return_sequences=True))
        model.add(Dropout(rate=0.2))
    
    model.add(LSTM(units=224)) 
    model.add(Dropout(rate=0.2))
    
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])
    return model

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
results = []

for train_index, val_index in kf.split(X_train_full):
    X_train, X_val = X_train_full[train_index], X_train_full[val_index]
    y_train, y_val = y_train_full.iloc[train_index], y_train_full.iloc[val_index]
    
    model = build_best_model()
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    model.fit(X_train, y_train, epochs=10, validation_split=0.2, callbacks=[early_stopping])
    
    loss, accuracy, precision, recall = model.evaluate(X_val, y_val)
    print(f"Fold {fold_no} - Validation Loss: {loss}, Validation Accuracy: {accuracy}, Validation Precision: {precision}, Validation Recall: {recall}")
    results.append((loss, accuracy, precision, recall))
    fold_no += 1

average_results = np.mean(results, axis=0)
print(f"Average Validation Loss: {average_results[0]}")
print(f"Average Validation Accuracy: {average_results[1]}")
print(f"Average Validation Precision: {average_results[2]}")
print(f"Average Validation Recall: {average_results[3]}")

final_model = build_best_model()
final_early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

final_model.fit(X_train_full, y_train_full, epochs=10, validation_split=0.2, callbacks=[final_early_stopping])

test_loss, test_accuracy, test_precision, test_recall = final_model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)

y_pred = final_model.predict(X_test)
y_pred_classes = np.where(y_pred > 0.5, 1, 0)

cm = confusion_matrix(y_test, y_pred_classes)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
