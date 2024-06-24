import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters

preprocessed_data_path_tokenized = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_train_data_tokenized.csv"
test_data_tokenized_path = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_test_data_tokenized.csv"

train_data_tokenized = pd.read_csv(preprocessed_data_path_tokenized)
test_data_tokenized = pd.read_csv(test_data_tokenized_path)

X_train_tokenized = train_data_tokenized.drop(columns=['oh_label'])
y_train_tokenized = train_data_tokenized['oh_label']

X_test_tokenized = test_data_tokenized.drop(columns=['oh_label'])
y_test_tokenized = test_data_tokenized['oh_label']

input_dim = X_train_tokenized.shape[1]

def build_model(hp):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim + 1,
                        output_dim=hp.Int('embedding_output_dim', min_value=32, max_value=256, step=32),
                        input_length=input_dim))
    num_layers = hp.Int('num_layers', 1, 3)
    for i in range(num_layers):
        model.add(GRU(units=hp.Int('gru_units_' + str(i), min_value=32, max_value=256, step=32), 
                      return_sequences=(i < num_layers - 1)))
        model.add(Dropout(rate=hp.Float('dropout_' + str(i), min_value=0.1, max_value=0.5, step=0.1)))
    activation_choice = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])
    model.add(Dense(1, activation=activation_choice))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])
    return model

tuner = RandomSearch(
    build_model,
    objective='val_recall',
    max_trials=5,
    executions_per_trial=2,
    directory='gru_hyperparameter_tuning',
    project_name='gru',
    overwrite=True
)

print("Starting hyperparameter tuning...")
tuner.search(X_train_tokenized, y_train_tokenized, epochs=6, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=3)], batch_size=32, verbose=1)
print("Hyperparameter tuning completed.")

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

def run_gru_with_cv_and_test(X_train, y_train, X_test, y_test, best_hps, method_name):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        model = build_model(best_hps)
        model.fit(X_train_fold, y_train_fold, epochs=10, validation_data=(X_val_fold, y_val_fold), 
                  callbacks=[EarlyStopping('val_loss', patience=3, restore_best_weights=True)], 
                  batch_size=32, verbose=0)
        
        val_loss, val_accuracy, val_precision, val_recall = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        accuracy_scores.append(val_accuracy)
        precision_scores.append(val_precision)
        recall_scores.append(val_recall)

    print(f'{method_name} - Mean Validation Accuracy: {np.mean(accuracy_scores)}, Mean Validation Precision: {np.mean(precision_scores)}, Mean Validation Recall: {np.mean(recall_scores)}')

    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test Precision: {test_precision}, Test Recall: {test_recall}')
    print()

run_gru_with_cv_and_test(X_train_tokenized, y_train_tokenized, X_test_tokenized, y_test_tokenized, best_hps, "Tokenized Data with Sentiment Analysis")
