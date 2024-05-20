import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.metrics import Precision, Recall
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

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

X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

def build_model(hp):
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=hp.Int('embedding_output_dim', min_value=32, max_value=256, step=32),
                        input_length=max_sequence_length))

    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(GRU(units=hp.Int('gru_units_' + str(i), min_value=32, max_value=256, step=32), return_sequences=(i < hp.Int('num_layers', 1, 3)-1)))
        model.add(Dropout(rate=hp.Float('dropout_' + str(i), min_value=0.1, max_value=0.5, step=0.1)))

    activation_choice = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])
    model.add(Dense(1, activation=activation_choice))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])

    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=2,
    directory='gru_hyperparameter_tuning',
    project_name='gru'
)

tuner.search(X_train, y_train, epochs=5, validation_split=0.2, callbacks=[EarlyStopping('val_loss', patience=3)], batch_size=32)

best_model = tuner.get_best_models(num_models=1)[0]

best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best Hyperparameters:")
print(best_hyperparameters.values)

loss, accuracy, precision, recall = best_model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
print("Test Precision:", precision)
print("Test Recall:", recall)
