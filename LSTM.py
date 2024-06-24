import pandas as pd
import numpy as np
import os
import shutil
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
from kerastuner.tuners import RandomSearch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

preprocessed_data_path_word2vec = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_train_data_word2vec.csv"
test_data_word2vec_path = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_test_data_word2vec.csv"

preprocessed_data_word2vec = pd.read_csv(preprocessed_data_path_word2vec)
test_data_word2vec = pd.read_csv(test_data_word2vec_path)

X_train_word2vec_with_sentiment = preprocessed_data_word2vec.iloc[:, :-1].values
y_train_word2vec = preprocessed_data_word2vec.iloc[:, -1].values

X_test_word2vec_with_sentiment = test_data_word2vec.iloc[:, :-1].values
y_test_word2vec = test_data_word2vec.iloc[:, -1].values

class_weights = compute_class_weight('balanced', classes=np.unique(y_train_word2vec), y=y_train_word2vec)
class_weights_dict = dict(enumerate(class_weights))

tuner_directory = 'lstm_hyperparameter_tuning'
if os.path.exists(tuner_directory):
    shutil.rmtree(tuner_directory)

def build_model(hp):
    model = Sequential()
    lstm_units = hp.Int('lstm_units', min_value=32, max_value=256, step=32)
    dropout_rate = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    activation_choice = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])
    num_layers = hp.Int('num_layers', 1, 3)
    model.add(LSTM(units=lstm_units, input_shape=(X_train_word2vec_with_sentiment.shape[1], 1), return_sequences=(num_layers > 1)))
    model.add(Dropout(rate=dropout_rate))
    for i in range(1, num_layers):
        model.add(LSTM(units=lstm_units, return_sequences=(i < num_layers - 1)))
        model.add(Dropout(rate=dropout_rate))
    model.add(Dense(1, activation=activation_choice))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory=tuner_directory,
    project_name='lstm'
)

def run_lstm_with_cv_and_test(X_train, y_train, X_test, y_test, method_name):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []

    X_train_expanded = np.expand_dims(X_train, axis=2)
    X_test_expanded = np.expand_dims(X_test, axis=2)

    tuner.search(X_train_expanded, y_train, epochs=10, validation_split=0.2, 
                 callbacks=[EarlyStopping('val_loss', patience=3, restore_best_weights=True)], 
                 class_weight=class_weights_dict,
                 batch_size=32, verbose=0)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = build_model(best_hps)

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        X_train_fold = np.expand_dims(X_train_fold, axis=2)
        X_val_fold = np.expand_dims(X_val_fold, axis=2)

        model.fit(X_train_fold, y_train_fold, epochs=10, validation_data=(X_val_fold, y_val_fold), 
                  callbacks=[EarlyStopping('val_loss', patience=3, restore_best_weights=True)], 
                  class_weight=class_weights_dict,
                  batch_size=32, verbose=0)
        
        val_loss, val_accuracy, val_precision, val_recall = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        accuracy_scores.append(val_accuracy)
        precision_scores.append(val_precision)
        recall_scores.append(val_recall)

    print(f'{method_name} - Mean Validation Accuracy: {np.mean(accuracy_scores)}, Mean Validation Precision: {np.mean(precision_scores)}, Mean Validation Recall: {np.mean(recall_scores)}')

    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test_expanded, y_test, verbose=0)
    y_pred = model.predict(X_test_expanded)
    y_pred = (y_pred > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test Precision: {test_precision}, Test Recall: {test_recall}')
    print()

run_lstm_with_cv_and_test(X_train_word2vec_with_sentiment, y_train_word2vec, X_test_word2vec_with_sentiment, y_test_word2vec, "Word2Vec with Sentiment Analysis")
