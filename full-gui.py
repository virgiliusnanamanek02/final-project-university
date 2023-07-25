import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def load_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        entry_file_path.delete(0, tk.END)
        entry_file_path.insert(0, file_path)

def analyze_data():
    file_path = entry_file_path.get()
    if not file_path:
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

    # Menggunakan data_test yang telah Anda berikan sebelumnya
    data_test = np.array([[10, 20, 30]])  # Ganti dengan nilai ppm dari masing-masing sensor gas
    hasil_prediksi = predict(data_test)

    # Membuat grafik probabilitas
    probabilitas = hasil_prediksi[0]

    # Daftar jenis kopi
    daftar_jenis_kopi = ['Kopi Sidikalang', 'Kopi Toraja', 'Kopi Temanggung', 'Kopi Manggarai']

    # Menemukan indeks dari probabilitas tertinggi
    indeks_tertinggi = np.argmax(probabilitas)

    # Menyiapkan warna
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

# Membuat GUI
root = tk.Tk()
root.title("Analisis Jenis Kopi")
root.geometry("400x300")

# Label dan Entry untuk file CSV
label_file_path = tk.Label(root, text="File CSV:")
label_file_path.pack()
entry_file_path = tk.Entry(root)
entry_file_path.pack()
btn_load_csv = tk.Button(root, text="Unggah File", command=load_csv)
btn_load_csv.pack()

# Label untuk suhu dan kelembaban (ganti dengan data sesuai kebutuhan)
label_temperature = tk.Label(root, text="Suhu: ...")
label_temperature.pack()
label_humidity = tk.Label(root, text="Kelembaban: ...")
label_humidity.pack()

# Tombol Analisis
btn_analyze = tk.Button(root, text="Analisis", command=analyze_data)
btn_analyze.pack()

# Menjalankan GUI
root.mainloop()