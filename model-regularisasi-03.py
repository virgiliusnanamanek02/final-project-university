import serial
import time
import tensorflow as tf
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt

# Buat koneksi dengan Arduino
ser = serial.Serial('COM3', 9600)

# Muat model yang sudah dilatih
model = tf.keras.models.load_model('./data/model-jst-11')

# Baca data pelatihan dari file CSV
data_training = pd.read_csv('merged.csv')

# Menghitung nilai minimum dan maksimum dari setiap kolom
X_min = data_training.min()
X_max = data_training.max()

# Nama-nama jenis kopi
jenis_kopi = ['Kopi Sidikalang', 'Kopi Toraja', 'Kopi Temanggung', 'Kopi Manggarai']

# Inisialisasi list untuk menyimpan data
data_buffer = []

# Jumlah data yang akan dikumpulkan sebelum menampilkan grafik
target_data_count = 100

# Threshold untuk klasifikasi
threshold = 50

while True:
    # Baca data dari Arduino
    data = ser.readline().decode('utf-8').strip()

    try:
        # Uraikan data JSON
        json_data = json.loads(data)

        # Ambil nilai mq7Ppm, mq135Ppm, dan mq136Ppm dari data JSON
        mq7Ppm = float(json_data["mq7Ppm"])
        mq135Ppm = float(json_data["mq135Ppm"])
        mq136Ppm = float(json_data["mq136Ppm"])

        # Normalisasi data masukan real-time
        mq7Ppm_normalized = (mq7Ppm - X_min['CO (MQ-7) PPM']) / (X_max['CO (MQ-7) PPM'] - X_min['CO (MQ-7) PPM'])
        mq135Ppm_normalized = (mq135Ppm - X_min['NH3 (MQ-135) PPM']) / (X_max['NH3 (MQ-135) PPM'] - X_min['NH3 (MQ-135) PPM'])
        mq136Ppm_normalized = (mq136Ppm - X_min['H2S (MQ-136) PPM']) / (X_max['H2S (MQ-136) PPM'] - X_min['H2S (MQ-136) PPM'])

        # Buat data input sesuai dengan model Anda (tanpa suhu dan kelembaban)
        input_data = np.array([[mq7Ppm_normalized, mq135Ppm_normalized, mq136Ppm_normalized]])

        # Klasifikasikan data dengan model
        predictions = model.predict(input_data)

        # Tambahkan data ke buffer
        data_buffer.append(predictions.ravel() * 100)

        # Jika sudah mencapai target_data_count, tampilkan grafik dan reset buffer
        if len(data_buffer) == target_data_count:
            # Gabungkan data dalam buffer menjadi satu array
            all_data = np.array(data_buffer)

            # Hitung rata-rata probabilitas untuk setiap jenis kopi
            average_predictions = np.mean(all_data, axis=0)

            # Tentukan hasil klasifikasi berdasarkan threshold
            if np.max(average_predictions) > threshold:
                predicted_class = jenis_kopi[np.argmax(average_prebdictions)]
                print(f'Jenis Kopi: {predicted_class}')
            else:
                print('Tidak dapat mengklasifikasikan jenis kopi.')

            # Reset buffer
            data_buffer = []

    except json.JSONDecodeError:
        print("Gagal menguraikan data JSON")

    # Beri sedikit jeda sebelum membaca data berikutnya
    time.sleep(1)
