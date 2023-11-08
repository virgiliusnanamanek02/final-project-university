import serial
import time
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
import threading

# Buat koneksi dengan Arduino
ser = serial.Serial('/dev/ttyUSB0', 9600)

# Muat model yang sudah dilatih
model = tf.keras.models.load_model('./data/model-jst')

# Inisialisasi list untuk menyimpan hasil probabilitas dan jenis kopi
probabilities = []
coffee_types = ["Kopi Sidikalang", "Kopi Toraja", "Kopi Temanggung", "Kopi Manggarai"]

# Inisialisasi plot
plt.ion()
fig, ax = plt.subplots()

# Counter untuk data uji
test_data_count = 0
max_test_data_count = 30  # Jumlah maksimal data uji yang ingin Anda ambil

# Buffer data
data_buffer = []

def read_data_from_arduino():
    global data_buffer
    while True:
        data = ser.readline().decode('utf-8').strip()
        data_buffer.append(data)

# Buat thread untuk membaca data dari Arduino
arduino_thread = threading.Thread(target=read_data_from_arduino)
arduino_thread.daemon = True
arduino_thread.start()

while test_data_count < max_test_data_count:
    if data_buffer:
        data = data_buffer.pop(0)

        try:
            # Uraikan data JSON
            json_data = json.loads(data)

            # Pastikan data JSON berisi nilai yang diperlukan
            if all(key in json_data for key in ["mq7Ppm", "mq135Ppm", "mq136Ppm"]):
                # Ambil nilai mq7Ppm, mq135Ppm, dan mq136Ppm dari data JSON
                mq7Ppm = float(json_data["mq7Ppm"])
                mq135Ppm = float(json_data["mq135Ppm"])
                mq136Ppm = float(json_data["mq136Ppm"])

                # Buat data input sesuai dengan model Anda (tanpa suhu dan kelembaban)
                input_data = np.array([[mq7Ppm, mq135Ppm, mq136Ppm]])

                # Klasifikasikan data dengan model
                predictions = model.predict(input_data)

                # Simpan hasil probabilitas
                probabilities.append(predictions[0])

                # Perbarui counter data uji
                test_data_count += 1

            else:
                print("Data JSON tidak lengkap")
        except json.JSONDecodeError:
            print("Gagal menguraikan data JSON")

# Setelah mengambil 30 data uji, gambar grafik
probabilities = probabilities[:len(coffee_types)]
ax.bar(coffee_types, [100 * p[0] for p in probabilities], alpha=0.75)
ax.set_ylim(0, 100)
ax.set_ylabel('Probabilitas (%)')
ax.set_title('Klasifikasi Kopi')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show(block=True)
