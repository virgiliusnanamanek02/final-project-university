import serial
import time
import threading
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt

# Inisialisasi list untuk menyimpan hasil probabilitas dan jenis kopi
probabilities = []
coffee_types = ["Kopi Sidikalang", "Kopi Toraja", "Kopi Temanggung", "Kopi Manggarai"]

# Inisialisasi plot
plt.ion()
fig, ax = plt.subplots()

# Counter untuk data uji
test_data_count = 0
max_test_data_count = 30  # Jumlah maksimal data uji yang ingin Anda ambil

# Fungsi untuk membaca data dari Arduino
def read_serial_data():
    ser = serial.Serial('/dev/ttyUSB0', 9600)
    global test_data_count, probabilities

    while test_data_count < max_test_data_count:
        data = ser.readline().decode('utf-8').strip()

        try:
            json_data = json.loads(data)
            if all(key in json_data for key in ["mq7Ppm", "mq135Ppm", "mq136Ppm"]):
                mq7Ppm = float(json_data["mq7Ppm"])
                mq135Ppm = float(json_data["mq135Ppm"])
                mq136Ppm = float(json_data["mq136Ppm"])
                input_data = np.array([[mq7Ppm, mq135Ppm, mq136Ppm]])
                predictions = model.predict(input_data)
                probabilities.append(predictions[0])
                test_data_count += 1
            else:
                print("Data JSON tidak lengkap")
        except json.JSONDecodeError:
            print("Gagal menguraikan data JSON")

# Mulai thread untuk membaca data serial
serial_thread = threading.Thread(target=read_serial_data)
serial_thread.start()

# Tunggu hingga data uji mencapai jumlah maksimum
serial_thread.join()

# Gambar grafik setelah mengambil semua data uji
probabilities = probabilities[:len(coffee_types)]
ax.bar(coffee_types, [100 * p[0] for p in probabilities], alpha=0.75)
ax.set_ylim(0, 100)
ax.set_ylabel('Probabilitas (%)')
ax.set_title('Klasifikasi Kopi')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show(block=True)
