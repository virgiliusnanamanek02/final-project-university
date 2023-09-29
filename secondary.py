import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PySimpleGUI as sg
from sklearn.metrics import accuracy_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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

    # Untuk memastikan hasil random yang sama setiap kali dijalankan
    np.random.seed(0)

    weights_input_hidden = np.random.uniform(
        size=(input_neurons, hidden_neurons))  # Bobot untuk input ke hidden layer
    bias_hidden = np.random.uniform(
        size=(1, hidden_neurons))  # Bias untuk hidden layer

    weights_hidden_output = np.random.uniform(
        size=(hidden_neurons, output_neurons))  # Bobot untuk hidden ke output layer
    bias_output = np.random.uniform(
        size=(1, output_neurons))  # Bias untuk output layer

    epochs = 10000  # Jumlah iterasi
    learning_rate = 0.1  # Nilai learning rate

    for _ in range(epochs):
        # Forward Propagation
        hidden_input = np.dot(X, weights_input_hidden) + \
            bias_hidden  # Input ke hidden layer
        hidden_output = sigmoid(hidden_input)  # Output dari hidden layer

        # Input ke output layer
        output_input = np.dot(
            hidden_output, weights_hidden_output) + bias_output
        predicted_output = sigmoid(output_input)  # Output dari output layer

        # Backpropagation
        error = y - predicted_output  # Selisih antara nilai target dengan nilai prediksi
        output_gradient = predicted_output * \
            (1 - predicted_output) * error  # Gradien dari output layer

        # Selisih antara nilai gradien output dengan nilai bobot hidden-output
        hidden_error = np.dot(output_gradient, weights_hidden_output.T)
        hidden_gradient = hidden_output * \
            (1 - hidden_output) * hidden_error  # Gradien dari hidden layer

        # Update bobot hidden-output
        weights_hidden_output += np.dot(hidden_output.T,
                                        output_gradient) * learning_rate
        # Update bias output
        bias_output += np.sum(output_gradient, axis=0,
                              keepdims=True) * learning_rate

        # Update bobot input-hidden
        weights_input_hidden += np.dot(X.T, hidden_gradient) * learning_rate
        # Update bias hidden
        bias_hidden += np.sum(hidden_gradient, axis=0,
                              keepdims=True) * learning_rate

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
    hidden_output = sigmoid(np.dot(X, weights_input_hidden) + bias_hidden)
    predicted_output = sigmoid(np.dot(
        hidden_output, weights_hidden_output) + bias_output)  # Output dari output layer

    # Membuat grafik probabilitas
    probabilitas = predicted_output[0]  # Probabilitas dari hasil prediksi

    # Daftar jenis kopi
    daftar_jenis_kopi = ['Kopi Sidikalang', 'Kopi Toraja',
                         'Kopi Temanggung', 'Kopi Manggarai']

    # Menemukan indeks dari probabilitas tertinggi
    # Indeks dari probabilitas tertinggi
    indeks_tertinggi = np.argmax(probabilitas)

    colors = ['gray'] * len(probabilitas)  # Menyiapkan warna
    colors[indeks_tertinggi] = 'green'  # Mewarnai probabilitas tertinggi

    # Menampilkan grafik hasil prediksi
    plt.figure(figsize=(8, 6))
    plt.bar(daftar_jenis_kopi, probabilitas, color=colors)
    plt.xlabel('Jenis Kopi')
    plt.ylabel('Probabilitas')
    plt.title('Probabilitas Jenis Kopi Berdasarkan Data')
    plt.ylim(0, 1)
    plt.show()


    # predicted_labels = (predicted_output >= 0.5).astype(int)
    accuracy = accuracy_score(np.argmax(y, axis=1), np.argmax(predicted_output, axis=1))
    # Menampilkan hasil evaluasi
    sg.popup(f'Akurasi: {accuracy * 100:.2f}%')
# Palet warna untuk GUI
sg.theme('DarkTeal2')

# Tampilan layout GUI untuk pelatihan dan pengujian dalam satu halaman
layout = [
    [sg.Text('Analisis Jenis Kopi', size=(30, 1), font=(
        'Helvetica', 20), justification='center')],
    [sg.Text('Pilih File Data Pelatihan:', size=(25, 1)), sg.Input(
        key='train_file_path', size=(40, 1)), sg.FileBrowse()],
    [sg.Text('Pilih File Data Pengujian:', size=(25, 1)), sg.Input(
        key='test_file_path', size=(40, 1)), sg.FileBrowse()],
    [sg.Button('Pelatihan', size=(20, 1)), sg.Button('Pengujian', size=(20, 1))],
    [sg.Text('Hasil Pelatihan/Pengujian:', size=(40, 1), justification='center')],
    [sg.Text('', key='result_text', size=(40, 1), justification='center')],
    [sg.Image(filename='', key='plot')],
    [sg.Text('Akurasi: ', key='accuracy_label', size=(15, 1)), sg.Text('', key='accuracy_value', size=(15, 1))],
    [sg.Button('Exit', size=(20, 1))],
]

# Membuat GUI window
window = sg.Window('Analisis Jenis Kopi', layout, size=(700, 500), element_justification='center')

weights_input_hidden, bias_hidden, weights_hidden_output, bias_output = None, None, None, None


accuracy = None
# Looping untuk event handling pada window
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED or event == 'Exit':
        break
    elif event == 'Pelatihan':
        # Panggil fungsi pelatihan dan perbarui hasilnya di GUI
        trained_model = train_model(values['train_file_path'])
        if trained_model:
            weights_input_hidden, bias_hidden, weights_hidden_output, bias_output = trained_model
            window['result_text'].update('Pelatihan selesai!')
    elif event == 'Pengujian':
        if weights_input_hidden is None:
            sg.popup_error('Lakukan pelatihan terlebih dahulu!')
        else:
            # Panggil fungsi pengujian dan perbarui hasilnya di GUI
            accuracy = test_model(values['test_file_path'], weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)
            window['result_text'].update('Pengujian selesai!')
            if accuracy is not None:
                window['accuracy_value'].update(f'{accuracy * 100:.2f}%')  # Perbarui akurasi jika tidak None
# Menutup window
window.close()
