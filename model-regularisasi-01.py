import os
import numpy as np
import pandas as pd
import PySimpleGUI as sg
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras import regularizers

save_path = "./data/model-jst-11"
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
    hidden_neurons = 32
    output_neurons = y.shape[1]

    np.random.seed(0)

    model = Sequential()
    model.add(Dense(hidden_neurons, input_dim=input_neurons, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(output_neurons, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))

    epochs = 500
    learning_rate = 0.01

    adagrad = Adagrad(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adagrad, metrics=['accuracy'])

    model.fit(X, y, epochs=epochs, batch_size=10)

    # Evaluasi model menggunakan metrik-metrik
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y, axis=1)

    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    sg.popup('Pelatihan sudah selesai!\n'
             f'Accuracy: {accuracy}\n'
             f'Precision: {precision}\n'
             f'Recall: {recall}\n'
             f'F1 Score: {f1}')

    if save_path:
        model.save(save_path)  # Simpan model ke lokasi yang telah ditentukan
        sg.popup('Model telah disimpan di: ' + save_path)

    return model


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
        model = train_model(training_path, save_path)

window.close()
