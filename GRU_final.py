import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Embedding
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

X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.1, random_state=42)

def build_best_model():
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=224, input_length=max_sequence_length))
    
    model.add(GRU(units=64, return_sequences=True))
    model.add(Dropout(rate=0.3))
    
    model.add(GRU(units=160, return_sequences=False))
    model.add(Dropout(rate=0.2))
    
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])
    
    return model

kf = KFold(n_splits=5, shuffle=True, random_state=42)
val_accuracy_list = []
val_precision_list = []
val_recall_list = []

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    model = build_best_model()
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    model.fit(X_train_fold, y_train_fold, epochs=5, validation_data=(X_val_fold, y_val_fold), callbacks=[early_stopping], batch_size=64)
    
    val_loss, val_accuracy, val_precision, val_recall = model.evaluate(X_val_fold, y_val_fold)
    val_accuracy_list.append(val_accuracy)
    val_precision_list.append(val_precision)
    val_recall_list.append(val_recall)

average_val_accuracy = np.mean(val_accuracy_list)
average_val_precision = np.mean(val_precision_list)
average_val_recall = np.mean(val_recall_list)

print("Average Validation Accuracy:", average_val_accuracy)
print("Average Validation Precision:", average_val_precision)
print("Average Validation Recall:", average_val_recall)

model = build_best_model()
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

model.fit(X_train, y_train, epochs=5, validation_split=0.2, callbacks=[early_stopping], batch_size=64)

loss, accuracy, precision, recall = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
print("Test Precision:", precision)
print("Test Recall:", recall)

y_pred = model.predict(X_test)
y_pred_classes = np.where(y_pred > 0.5, 1, 0)

cm = confusion_matrix(y_test, y_pred_classes)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
