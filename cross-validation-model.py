import os
import numpy as np
import pandas as pd
import PySimpleGUI as sg
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras import regularizers
from scikeras.wrappers import KerasClassifier

save_path = "./data/model-jst-04"

# Fungsi untuk membangun model Keras dengan parameter yang dapat dioptimalkan
def build_model(neurons=16, dropout_rate=0.2, l2_penalty=0.01, learning_rate=0.01):
    input_neurons = 3
    output_neurons = 4

    model = Sequential()
    model.add(Dense(neurons, input_dim=input_neurons, activation='relu', kernel_regularizer=regularizers.l2(l2_penalty)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(output_neurons, activation='softmax', kernel_regularizer=regularizers.l2(l2_penalty)))

    adagrad = Adagrad(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adagrad, metrics=['accuracy'])

    return model

# Fungsi wrapper untuk model Keras agar bisa digunakan dengan alat-alat Scikit-Learn
def create_keras_model(neurons=16, dropout_rate=0.2, l2_penalty=0.01, learning_rate=0.01):
    input_neurons = 3
    output_neurons = 4

    model = Sequential()
    model.add(Dense(neurons, input_dim=input_neurons, activation='relu', kernel_regularizer=regularizers.l2(l2_penalty)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(output_neurons, activation='softmax', kernel_regularizer=regularizers.l2(l2_penalty)))

    adagrad = Adagrad(learning_rate=learning_rate)  # Pass the learning_rate parameter to the Adagrad optimizer
    model.compile(loss='categorical_crossentropy', optimizer=adagrad, metrics=['accuracy'])

    return model

# Fungsi untuk melatih model dengan optimasi hyperparameter
def train_model(file_path, save_path):
    if not file_path:
        sg.popup_error('Pilih file data pelatihan terlebih dahulu!')
        return None

    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() != '.csv':
        sg.popup_error('File harus berekstensi CSV!')
        return None

    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()
    X = data[['CO (MQ-7) PPM', 'NH3 (MQ-135) PPM', 'H2S (MQ-136) PPM']].values
    y = data[['Kopi Sidikalang', 'Kopi Toraja',
              'Kopi Temanggung', 'Kopi Manggarai']].values

    # Membuat model KerasClassifier yang dapat digunakan dengan Scikit-Learn
    model = KerasClassifier(build_fn=create_keras_model, neurons=16, verbose=0)

    # Definisi hyperparameter yang akan diuji
    param_dist = {
        'dropout_rate': [0.2, 0.4, 0.6],
        'l2_penalty': [0.001, 0.01, 0.1],
        'learning_rate': [0.001, 0.01, 0.1]
    }

    # Metrik evaluasi
    scoring = make_scorer(accuracy_score)

    # RandomizedSearchCV untuk mencari kombinasi hyperparameter terbaik
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, scoring=scoring, cv=5)
    random_search.fit(X, y)

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print(f'Best Parameters: {best_params}')
    print(f'Best Accuracy: {best_score}')

    sg.popup('Pelatihan sudah selesai!\n'
             f'Best Parameters: {best_params}\n'
             f'Best Accuracy: {best_score}')

    if save_path:
        best_model.model.save(save_path)  # Simpan model ke lokasi yang telah ditentukan

# GUI untuk memilih file CSV pelatihan
layout = [[sg.Text('Pilih File CSV Pelatihan')],
          [sg.Input(key='TRAINING_FILE'), sg.FileBrowse(file_types=(("CSV Files", "*.csv"),))],
          [sg.Button('Train Model'), sg.Button('Exit')]]

window = sg.Window('Train Model GUI', layout)

while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    elif event == 'Train Model':
        training_path = values['TRAINING_FILE']
        train_model(training_path, save_path)

window.close()
