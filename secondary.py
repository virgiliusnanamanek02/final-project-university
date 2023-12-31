import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PySimpleGUI as sg
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

# Fungsi untuk melatih model
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
    hidden_neurons = 8
    output_neurons = y.shape[1]

    np.random.seed(0)

    weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
    bias_hidden = np.random.uniform(size=(1, hidden_neurons))

    weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))
    bias_output = np.random.uniform(size=(1, output_neurons))

    epochs = 500
    learning_rate = 0.1

    for _ in range(epochs):
        hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
        hidden_output = sigmoid(hidden_input)

        output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
        predicted_output = softmax(output_input)

        error = y - predicted_output
        output_gradient = predicted_output * (1 - predicted_output) * error

        hidden_error = np.dot(output_gradient, weights_hidden_output.T)
        hidden_gradient = hidden_output * (1 - hidden_output) * hidden_error

        weights_hidden_output += np.dot(hidden_output.T, output_gradient) * learning_rate
        bias_output += np.sum(output_gradient, axis=0, keepdims=True) * learning_rate

        weights_input_hidden += np.dot(X.T, hidden_gradient) * learning_rate
        bias_hidden += np.sum(hidden_gradient, axis=0, keepdims=True) * learning_rate

    sg.popup('Pelatihan sudah selesai!')
    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

# Fungsi untuk menguji model
def test_model(file_path, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
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

    hidden_output = sigmoid(np.dot(X, weights_input_hidden) + bias_hidden)
    predicted_output = softmax(np.dot(hidden_output, weights_hidden_output) + bias_output)

    true_labels = np.argmax(y, axis=1)
    accuracy = accuracy_score(true_labels, np.argmax(predicted_output, axis=1))
    precision = precision_score(true_labels, np.argmax(predicted_output, axis=1), average='weighted', zero_division=0)
    recall = recall_score(true_labels, np.argmax(predicted_output, axis=1), average='weighted', zero_division=0)

    f1 = f1_score(true_labels, np.argmax(predicted_output, axis=1), average='weighted')

    print(f'Akurasi: {accuracy * 100:.2f}%')
    print(f'Presisi: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1-Score: {f1 * 100:.2f}%')

    #sg.popup(f'Accuracy: {accuracy * 100:.2f}%', f'Precision: {precision * 100:.2f}%', f'Recall: {recall * 100:.2f}%', f'F1-Score: {f1 * 100:.2f}%')

    probabilitas = predicted_output  # Probabilitas tidak perlu diambil sesuai dengan kategori, Anda dapat membiarkannya seperti ini.

    # Pastikan daftar_jenis_kopi memiliki jumlah yang sesuai dengan data pengujian.
    daftar_jenis_kopi = ['Kopi Sidikalang', 'Kopi Toraja', 'Kopi Temanggung', 'Kopi Manggarai']
    bar_colors = ['gray'] * len(probabilitas)
    for i in range(len(bar_colors)):
        bar_colors[i] = 'green' if np.argmax(probabilitas, axis=1)[i] == true_labels[i] else 'gray'

    plt.figure(figsize=(8, 6))
    plt.bar(daftar_jenis_kopi, probabilitas.mean(axis=0), color=bar_colors)  # Ambil rata-rata probabilitas
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

weights_input_hidden, bias_hidden, weights_hidden_output, bias_output = None, None, None, None

accuracy = None

while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED or event == 'Exit':
        break
    elif event == 'Pelatihan':
        trained_model = train_model(values['train_file_path'])
        if trained_model:
            weights_input_hidden, bias_hidden, weights_hidden_output, bias_output = trained_model
            window['result_text'].update('Pelatihan selesai!')
    elif event == 'Pengujian':
        if weights_input_hidden is None:
            sg.popup_error('Lakukan pelatihan terlebih dahulu!')
        else:
            accuracy = test_model(values['test_file_path'], weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)
            window['result_text'].update('Pengujian selesai!')
            if accuracy is not None:
                window['accuracy_value'].update(f'{accuracy * 100:.2f}%')

window.close()

