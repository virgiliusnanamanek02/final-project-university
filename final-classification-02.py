import serial
import time
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt

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

while True:
    # Baca data dari Arduino
    data = ser.readline().decode('utf-8').strip()

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

            # Jika sudah ada cukup data, gambar grafik
            if len(probabilities) >= len(coffee_types):
                ax.clear()
                ax.bar(coffee_types, [100 * p[0] for p in probabilities], alpha=0.75)
                ax.set_ylim(0, 100)
                ax.set_ylabel('Probabilitas (%)')
                ax.set_title('Klasifikasi Kopi')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.pause(0.1)

        else:
            print("Data JSON tidak lengkap")
    except json.JSONDecodeError:
        print("Gagal menguraikan data JSON")

    # Beri sedikit jeda sebelum membaca data berikutnya
    time.sleep(1)
