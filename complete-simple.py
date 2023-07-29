import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PySimpleGUI as sg

def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def load_csv(file_key):
    file_path = sg.popup_get_file('Pilih File CSV', file_types=(("CSV Files", "*.csv"),)) 
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
    X = data[['CO (MQ-7) PPM', 'NH3 (MQ-135) PPM', 'H2S (MQ-136) PPM']].values
    y = data[['Kopi Sidikalang', 'Kopi Toraja', 'Kopi Temanggung', 'Kopi Manggarai']].values

    input_neurons = X.shape[1] # 3 karena ada 3 sensor gas
    hidden_neurons = 4
    output_neurons = y.shape[1] # 4 karena ada 4 jenis kopi

    np.random.seed(0) # Untuk memastikan hasil random yang sama setiap kali dijalankan

    weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons)) # Bobot untuk input ke hidden layer
    bias_hidden = np.random.uniform(size=(1, hidden_neurons)) # Bias untuk hidden layer

    weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons)) # Bobot untuk hidden ke output layer
    bias_output = np.random.uniform(size=(1, output_neurons)) # Bias untuk output layer

    epochs = 10000 # Jumlah iterasi
    learning_rate = 0.1 # Nilai learning rate

    for _ in range(epochs):
        # Forward Propagation
        hidden_input = np.dot(X, weights_input_hidden) + bias_hidden # Input ke hidden layer
        hidden_output = sigmoid(hidden_input) # Output dari hidden layer

        output_input = np.dot(hidden_output, weights_hidden_output) + bias_output # Input ke output layer
        predicted_output = sigmoid(output_input) # Output dari output layer

        # Backpropagation
        error = y - predicted_output # Selisih antara nilai target dengan nilai prediksi
        output_gradient = predicted_output * (1 - predicted_output) * error # Gradien dari output layer

        hidden_error = np.dot(output_gradient, weights_hidden_output.T) # Selisih antara nilai gradien output dengan nilai bobot hidden-output
        hidden_gradient = hidden_output * (1 - hidden_output) * hidden_error # Gradien dari hidden layer

        weights_hidden_output += np.dot(hidden_output.T, output_gradient) * learning_rate # Update bobot hidden-output
        bias_output += np.sum(output_gradient, axis=0, keepdims=True) * learning_rate # Update bias output

        weights_input_hidden += np.dot(X.T, hidden_gradient) * learning_rate # Update bobot input-hidden
        bias_hidden += np.sum(hidden_gradient, axis=0, keepdims=True) * learning_rate # Update bias hidden

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
    y = data[['Kopi Sidikalang', 'Kopi Toraja', 'Kopi Temanggung', 'Kopi Manggarai']].values

    hidden_output = sigmoid(np.dot(X, weights_input_hidden) + bias_hidden) # Output dari hidden layer
    predicted_output = sigmoid(np.dot(hidden_output, weights_hidden_output) + bias_output) # Output dari output layer

    # Membuat grafik probabilitas
    probabilitas = predicted_output[0] # Probabilitas dari hasil prediksi

    # Daftar jenis kopi
    daftar_jenis_kopi = ['Kopi Sidikalang', 'Kopi Toraja', 'Kopi Temanggung', 'Kopi Manggarai'] 

    # Menemukan indeks dari probabilitas tertinggi
    indeks_tertinggi = np.argmax(probabilitas) # Indeks dari probabilitas tertinggi

    colors = ['gray'] * len(probabilitas) # Menyiapkan warna
    colors[indeks_tertinggi] = 'green' # Mewarnai probabilitas tertinggi

    # Menampilkan grafik hasil prediksi
    plt.figure(figsize=(8, 6)) 
    plt.bar(daftar_jenis_kopi, probabilitas, color=colors)
    plt.xlabel('Jenis Kopi')
    plt.ylabel('Probabilitas')
    plt.title('Probabilitas Jenis Kopi Berdasarkan Data')
    plt.ylim(0, 1)
    plt.show()

# Palet warna untuk GUI
sg.theme('DarkTeal2')

# Tampilan layout GUI untuk pelatihan
train_layout = [
    [sg.Text('Analisis Jenis Kopi', size=(30,1), font=('Helvetica', 20), justification='center')],
    [sg.Text('Pilih File Data Pelatihan:', size=(25, 1)), sg.Input(key='train_file_path', size=(40, 1)), sg.FileBrowse()],
    [sg.Button('Pelatihan', size=(20, 1)),
     sg.Text('Tutup untuk melanjutkan ke tahap pengujian', size=(20,1), justification='center')],
]

# Membuat GUI window untuk pelatihan
train_window = sg.Window('Pelatihan Model', train_layout, size=(700, 150), element_justification='center')

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
        [sg.Text('Analisis Jenis Kopi', size=(30,1), font=('Helvetica', 20), justification='center')],
        [sg.Text('Pilih File Data Pengujian:', size=(25, 1)), sg.Input(key='test_file_path', size=(40, 1)), sg.FileBrowse()],
        [sg.Button('Pengujian', size=(20, 1)), sg.Button('Exit', size=(20, 1))],
        [sg.Text('Suhu: ...', key='temperature', pad=(20, 0))],
        [sg.Text('Kelembaban: ...', key='humidity')],
        [sg.Image(filename='', key='plot')],
    ]

    # Membuat GUI window untuk pengujian
    test_window = sg.Window('Pengujian Model', test_layout, size=(700, 500), element_justification='center')

    # Looping untuk event handling pada window pengujian
    while True:
        event, values = test_window.read()
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break
        elif event == 'Pengujian':
            test_model(values['test_file_path'], weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)

        # Update data suhu dan kelembaban secara acak untuk contoh
        suhu_acak = np.random.randint(20, 30)
        kelembaban_acak = np.random.randint(50, 80)
        test_window['temperature'].update(f'Suhu: {suhu_acak} °C')
        test_window['humidity'].update(f'Kelembaban: {kelembaban_acak} %RH')

    # Menutup window pengujian
    test_window.close()
