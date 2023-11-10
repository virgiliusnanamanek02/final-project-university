import os
import numpy as np
import pandas as pd
import PySimpleGUI as sg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras import regularizers

save_path = "./data/model-jst-04"
training_path = "merged.csv"


# Fungsi untuk memuat file CSV
def load_csv(file_key):
    file_path = sg.popup_get_file(
        'Pilih File CSV', file_types=(("CSV Files", "*.csv"),))
    if file_path:
        sg.window[file_key].update(file_path)

# Fungsi untuk melatih model dengan regularisasi
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

    input_neurons = X.shape[1]
    hidden_neurons = 16
    output_neurons = y.shape[1]

    np.random.seed(0)

    model = Sequential()
    model.add(Dense(hidden_neurons, input_dim=input_neurons, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(output_neurons, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))

    epochs = 500
    learning_rate = 0.1

    adagrad = Adagrad(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adagrad, metrics=['accuracy'])

    model.fit(X, y, epochs=epochs, batch_size=10)

    sg.popup('Pelatihan sudah selesai!')

    if save_path:
        model.save(save_path)  # Simpan model ke lokasi yang telah ditentukan
        sg.popup('Model telah disimpan di: ' + save_path)

    return model

model = train_model(training_path, save_path)

print(model)
