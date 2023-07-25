import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PySimpleGUI as sg

def load_csv():
    file_path = sg.popup_get_file('Pilih File CSV', file_types=(("CSV Files", "*.csv"),))
    if file_path:
        window['file_path'].update(file_path)

def analyze_data():
    file_path = values['file_path']
    if not file_path:
        sg.popup_error('Pilih file CSV terlebih dahulu!')
        return

    data = pd.read_csv(file_path)
    X = data[['CO (MQ-7) PPM', 'NH3 (MQ-135) PPM', 'H2S (MQ-136) PPM']].values
    y = data[['Kopi Sidikalang', 'Kopi Toraja', 'Kopi Temanggung', 'Kopi Manggarai']].values

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    input_neurons = X.shape[1]
    hidden_neurons = 4
    output_neurons = y.shape[1]

    np.random.seed(0)

    weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
    bias_hidden = np.random.uniform(size=(1, hidden_neurons))

    weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))
    bias_output = np.random.uniform(size=(1, output_neurons))

    epochs = 10000
    learning_rate = 0.1

    for epoch in range(epochs):
        # Forward Propagation
        hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
        hidden_output = sigmoid(hidden_input)

        output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
        predicted_output = sigmoid(output_input)

        # Backpropagation
        error = y - predicted_output
        output_gradient = predicted_output * (1 - predicted_output) * error

        hidden_error = np.dot(output_gradient, weights_hidden_output.T)
        hidden_gradient = hidden_output * (1 - hidden_output) * hidden_error

        weights_hidden_output += np.dot(hidden_output.T, output_gradient) * learning_rate
        bias_output += np.sum(output_gradient, axis=0, keepdims=True) * learning_rate

        weights_input_hidden += np.dot(X.T, hidden_gradient) * learning_rate
        bias_hidden += np.sum(hidden_gradient, axis=0, keepdims=True) * learning_rate

    def predict(X):
        hidden_output = sigmoid(np.dot(X, weights_input_hidden) + bias_hidden)
        predicted_output = sigmoid(np.dot(hidden_output, weights_hidden_output) + bias_output)
        return predicted_output
    
    # ... Bagian Kode Analisis Data seperti sebelumnya ...
    data_test = np.array([[10, 20, 30]])  # Ganti dengan nilai ppm dari masing-masing sensor gas
    hasil_prediksi = predict(data_test)

    # Membuat grafik probabilitas
    probabilitas = hasil_prediksi[0]

    # Daftar jenis kopi
    daftar_jenis_kopi = ['Kopi Sidikalang', 'Kopi Toraja', 'Kopi Temanggung', 'Kopi Manggarai']

# Menemukan indeks dari probabilitas tertinggi
    indeks_tertinggi = np.argmax(probabilitas)

    colors = ['gray'] * len(probabilitas)
    colors[indeks_tertinggi] = 'orange'


    # Menampilkan grafik
    plt.figure(figsize=(8, 6))
    plt.bar(daftar_jenis_kopi, probabilitas, color=colors)
    plt.xlabel('Jenis Kopi')
    plt.ylabel('Probabilitas')
    plt.title('Probabilitas Jenis Kopi Berdasarkan Data')
    plt.ylim(0, 1)
    plt.show()

# Palet warna untuk GUI
sg.theme('DarkGrey5')

# Tampilan layout GUI
layout = [
    [sg.Text('File CSV:'), sg.Input(key='file_path'), sg.FileBrowse()],
    [sg.Button('Analisis'), sg.Button('Exit')],
    [sg.Text('Suhu: ...', key='temperature')],
    [sg.Text('Kelembaban: ...', key='humidity')],
    [sg.Image(filename='', key='plot')],
]

# Membuat GUI window
window = sg.Window('Analisis Jenis Kopi', layout, size=(700, 500), element_justification='center')

# Looping untuk event handling
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED or event == 'Exit':
        break
    elif event == 'Analisis':
        analyze_data()
    elif event == 'Browse':
        load_csv()

    # Update data suhu dan kelembaban secara acak untuk contoh
    suhu_acak = np.random.randint(20, 30)
    kelembaban_acak = np.random.randint(50, 80)
    window['temperature'].update(f'Suhu: {suhu_acak} °C')
    window['humidity'].update(f'Kelembaban: {kelembaban_acak} %RH')

# Menutup GUI window
window.close()