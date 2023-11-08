import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PySimpleGUI as sg
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adagrad

# Fungsi sigmoid untuk aktivasi
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Fungsi softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0, keepdims=True)

# Fungsi untuk memuat file CSV
def load_csv(file_key):
    file_path = sg.popup_get_file(
        'Pilih File CSV', file_types=(("CSV Files", "*.csv"),))
    if file_path:
        sg.window[file_key].update(file_path)

# Fungsi untuk melatih model dengan dropout
def train_model(file_path):
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
    model.add(Dense(hidden_neurons, input_dim=input_neurons, activation='relu'))
    model.add(Dropout(0.2))  # Dropout layer with a 20% dropout rate
    model.add(Dense(output_neurons, activation='softmax'))

    epochs = 500
    learning_rate = 0.1

    #sgd = SGD(lr=learning_rate)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    #rmsprop = RMSprop(learning_rate=learning_rate)
    #model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

    adagrad = Adagrad(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adagrad, metrics=['accuracy'])


    model.fit(X, y, epochs=epochs, batch_size=10)

    sg.popup('Pelatihan sudah selesai!')
    return model

# Fungsi untuk menguji model dengan dropout
def test_model(file_path, model):
    if not file_path:
        sg.popup_error('Pilih file data pengujian terlebih dahulu!')
        return

    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() != '.csv':
        sg.popup_error('File harus berekstensi CSV!')
        return

    data = pd.read_csv(file_path)
    X = data[['CO (MQ-7) PPM', 'NH3 (MQ-135) PPM', 'H2S (MQ-136) PPM']].values
    y = data[['Kopi Sidikalang', 'Kopi Toraja',
              'Kopi Temanggung', 'Kopi Manggarai']].values

    predicted_output = model.predict(X)

    true_labels = np.argmax(y, axis=1)

    accuracy = accuracy_score(true_labels, np.argmax(predicted_output, axis=1))
    precision = precision_score(true_labels, np.argmax(predicted_output, axis=1), average='weighted', zero_division=0)

    # Hitung recall
    conf_matrix = confusion_matrix(true_labels, np.argmax(predicted_output, axis=1))
    recall = recall_score(true_labels, np.argmax(predicted_output, axis=1), average=None)
    
    # Hitung weighted average recall
    if np.isnan(precision):
        weighted_recall = np.nan
    else:
        weighted_recall = np.sum(np.diag(recall) / np.sum(conf_matrix, axis=1))

    # Hitung F1-Score
    if np.isnan(precision) or np.isnan(weighted_recall):
        f1 = np.nan
    else:
        f1 = 2 * (precision * weighted_recall) / (precision + weighted_recall)

    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    
    if np.isnan(weighted_recall):
        print('Recall: Not defined (no true samples in some labels)')
    else:
        print(f'Recall: {weighted_recall * 100:.2f}%')

    if np.isnan(f1):
        print('F1-Score: Not defined (precision or recall is undefined)')
    else:
        print(f'F1-Score: {f1 * 100:.2f}%')

   # sg.popup(f'Accuracy: {accuracy * 100:.2f}%', f'Precision: {precision * 100:.2f}%', f'Recall: {recall * 100:.2f}%', f'F1-Score: {f1 * 100:.2f}%')

    probabilitas = predicted_output

    daftar_jenis_kopi = ['Kopi Sidikalang', 'Kopi Toraja', 'Kopi Temanggung', 'Kopi Manggarai']
    bar_colors = ['gray'] * len(probabilitas)
    for i in range(len(bar_colors)):
        bar_colors[i] = 'green' if np.argmax(probabilitas, axis=1)[i] == true_labels[i] else 'gray'

    plt.figure(figsize=(8, 6))
    plt.bar(daftar_jenis_kopi, probabilitas.mean(axis=0), color=bar_colors)
    plt.xlabel('Jenis Kopi')
    plt.ylabel('Probabilitas')
    plt.title('Probabilitas Jenis Kopi Berdasarkan Data Uji')
    plt.ylim(0, 1)
    plt.show()

    conf_matrix = confusion_matrix(true_labels, np.argmax(predicted_output, axis=1))

    print("Matriks Konfusi:")
    print(conf_matrix)


sg.theme('SandyBeach')

layout = [
    [sg.Text('Analisis Jenis Kopi', size=(30, 1), font=('Helvetica', 20), justification='center')],
    [sg.Text('Pilih File Data Pelatihan:', size=(25, 1)), sg.Input(key='train_file_path', size=(40, 1)), sg.FileBrowse()],
    [sg.Text('Pilih File Data Pengujian:', size=(25, 1)), sg.Input(key='test_file_path', size=(40, 1)), sg.FileBrowse()],
    [sg.Button('Pelatihan', size=(20, 1)), sg.Button('Pengujian', size=(20, 1))],
    [sg.Text('Hasil Pelatihan/Pengujian:', size=(40, 1), justification='center')],
    [sg.Text('', key='result_text', size=(40, 1), justification='center')],
    [sg.Image(filename='', key='plot')],
    [sg.Button('Exit', size=(20, 1))],
]

window = sg.Window('Analisis Jenis Kopi', layout, size=(600, 300), element_justification='center')

model = None
accuracy = None

while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED or event == 'Exit':
        break
    elif event == 'Pelatihan':
        model = train_model(values['train_file_path'])
        if model:
            window['result_text'].update('Pelatihan selesai!')
    elif event == 'Pengujian':
        if model is None:
            sg.popup_error('Lakukan pelatihan terlebih dahulu!')
        else:
            accuracy = test_model(values['test_file_path'], model)
            window['result_text'].update('Pengujian selesai!')
            if accuracy is not None:
                window['accuracy_value'].update(f'{accuracy * 100:.2f}%')

window.close()
