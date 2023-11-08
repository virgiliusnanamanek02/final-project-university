
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PySimpleGUI as sg
from sklearn.metrics import accuracy_score


def relu(x):
    return np.maximum(0, x)


def initialize_weights(input_neurons, hidden_neurons, output_neurons):
    # Inisialisasi bobot input-hidden dengan metode He
    weights_input_hidden = np.random.randn(
        input_neurons, hidden_neurons) * np.sqrt(2.0 / input_neurons)

    # Inisialisasi bobot hidden-output dengan metode He
    weights_hidden_output = np.random.randn(
        hidden_neurons, output_neurons) * np.sqrt(2.0 / hidden_neurons)

    return weights_input_hidden, weights_hidden_output


def load_csv(file_key):
    file_path = sg.popup_get_file(
        'Pilih File CSV', file_types=(("CSV Files", "*.csv"),))
    if file_path:
        sg.window[file_key].update(file_path)


def train_model(file_path):
    if not file_path:
        sg.popup_error('Pilih file data pelatihan terlebih dahulu!')
        return None

    # Memeriksa ekstensi file
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() != '.csv':
        sg.popup_error('File harus berekstensi CSV!')
        return None

    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()
    X = data[['CO (MQ-7) PPM', 'NH3 (MQ-135) PPM', 'H2S (MQ-136) PPM']].values
    y = data[['Kopi Sidikalang', 'Kopi Toraja',
              'Kopi Temanggung', 'Kopi Manggarai']].values

    input_neurons = X.shape[1]  # 3 karena ada 3 sensor gas
    hidden_neurons = 4
    output_neurons = y.shape[1]  # 4 karena ada 4 jenis kopi

    # Inisialisasi bobot dengan metode He
    weights_input_hidden, weights_hidden_output = initialize_weights(
        input_neurons, hidden_neurons, output_neurons)

    bias_hidden = np.random.randn(1, hidden_neurons)  # Bias untuk hidden layer
    bias_output = np.random.randn(1, output_neurons)  # Bias untuk output layer

    epochs = 10000  # Jumlah iterasi
    learning_rate = 0.01  # Nilai learning rate

    for _ in range(epochs):
        for i in range(len(X)):
            # Forward Propagation
            hidden_input = np.dot(X[i], weights_input_hidden) + bias_hidden
            hidden_output = relu(hidden_input)

            output_input = np.dot(
                hidden_output, weights_hidden_output) + bias_output
            predicted_output = output_input

            # Backpropagation

            error = y[i] - predicted_output
            output_gradient = error
            hidden_error = np.dot(output_gradient, weights_hidden_output.T)
            hidden_gradient = (hidden_error > 0).astype(int) * hidden_error

            # Update bobot dan bias dengan SGD
            weights_hidden_output += np.outer(hidden_output,
                                              output_gradient) * learning_rate
            bias_output += output_gradient * learning_rate

            weights_input_hidden += np.outer(X[i],
                                             hidden_gradient) * learning_rate
            bias_hidden += hidden_gradient * learning_rate

    sg.popup('Pelatihan sudah selesai!')

    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output


def test_model(file_path, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    if not file_path:
        sg.popup_error('Pilih file data pengujian terlebih dahulu!')
        return

    # Memeriksa ekstensi file
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() != '.csv':
        sg.popup_error('File harus berekstensi CSV!')
        return

    data = pd.read_csv(file_path)
    X = data[['CO (MQ-7) PPM', 'NH3 (MQ-135) PPM', 'H2S (MQ-136) PPM']].values
    y = data[['Kopi Sidikalang', 'Kopi Toraja',
              'Kopi Temanggung', 'Kopi Manggarai']].values

    # Output dari hidden layer
    hidden_output = relu(np.dot(X, weights_input_hidden) + bias_hidden)
    predicted_output = np.dot(
        hidden_output, weights_hidden_output) + bias_output

    # Membuat grafik probabilitas
    probabilitas = predicted_output[0]

    # Daftar jenis kopi
    daftar_jenis_kopi = ['Kopi Sidikalang', 'Kopi Toraja',
                         'Kopi Temanggung', 'Kopi Manggarai']

    # Menemukan indeks dari probabilitas tertinggi
    indeks_tertinggi = np.argmax(probabilitas)

    colors = ['gray'] * len(probabilitas)
    colors[indeks_tertinggi] = 'green'

    # Menampilkan grafik hasil prediksi
    plt.figure(figsize=(8, 6))
    plt.bar(daftar_jenis_kopi, probabilitas, color=colors)
    plt.xlabel('Jenis Kopi')
    plt.ylabel('Probabilitas')
    plt.title('Probabilitas Jenis Kopi Berdasarkan Data')
    plt.ylim(0, 1)
    plt.show()

    accuracy = accuracy_score(np.argmax(y, axis=1),
                              np.argmax(predicted_output, axis=1))
    sg.popup(f'Akurasi: {accuracy * 100:.2f}%')


# Palet warna untuk GUI
sg.theme('DarkTeal2')

# Tampilan layout GUI untuk pelatihan
train_layout = [
    [sg.Text('Analisis Jenis Kopi', size=(30, 1), font=(
        'Helvetica', 20), justification='center')],
    [sg.Text('Pilih File Data Pelatihan:', size=(25, 1)), sg.Input(
        key='train_file_path', size=(40, 1)), sg.FileBrowse()],
    [sg.Button('Pelatihan', size=(20, 1))],
    [sg.Text('Tutup untuk melanjutkan ke tahap pengujian',
             size=(40, 1), justification='center')]
]

# Membuat GUI window untuk pelatihan
train_window = sg.Window('Pelatihan Model', train_layout, size=(
    700, 150), element_justification='center')

# Looping untuk event handling pada window pelatihan
while True:
    event, values = train_window.read()
    if event == sg.WINDOW_CLOSED:
        break
    elif event == 'Pelatihan':
        trained_model = train_model(values['train_file_path'])

# Menutup window pelatihan
train_window.close()

# Jika pelatihan selesai, lanjut ke tahap pengujian
if trained_model:
    weights_input_hidden, bias_hidden, weights_hidden_output, bias_output = trained_model

    # Tampilan layout GUI untuk pengujian
    test_layout = [
        [sg.Text('Analisis Jenis Kopi', size=(30, 1), font=(
            'Helvetica', 20), justification='center')],
        [sg.Text('Pilih File Data Pengujian:', size=(25, 1)), sg.Input(
            key='test_file_path', size=(40, 1)), sg.FileBrowse()],
        [sg.Button('Pengujian', size=(20, 1)),
         sg.Button('Exit', size=(20, 1))],
        [sg.Text('Suhu: ...', key='temperature', pad=(20, 0))],
        [sg.Text('Kelembaban: ...', key='humidity')],
        [sg.Image(filename='', key='plot')],
    ]

    # Membuat GUI window untuk pengujian
    test_window = sg.Window('Pengujian Model', test_layout, size=(
        700, 500), element_justification='center')

    # Looping untuk event handling pada window pengujian
    while True:
        event, values = test_window.read()
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break
        elif event == 'Pengujian':
            test_model(values['test_file_path'], weights_input_hidden,
                       bias_hidden, weights_hidden_output, bias_output)

    # Menutup window pengujian
    test_window.close()
