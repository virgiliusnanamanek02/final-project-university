import serial
import time
import tensorflow as tf
import numpy as np
import json

# Buat koneksi dengan Arduino
ser = serial.Serial('/dev/ttyUSB0', 9600) 

# Muat model yang sudah dilatih
model = tf.keras.models.load_model('./data/model-jst')

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
        
        # Buat data input sesuai dengan model Anda (tanpa suhu dan kelembaban)
        input_data = np.array([[mq7Ppm, mq135Ppm, mq136Ppm]])
        
        # Klasifikasikan data dengan model
        predictions = model.predict(input_data)
        
        # Temukan indeks kategori dengan probabilitas tertinggi
        predicted_category = np.argmax(predictions, axis=1)
        
        # Tampilkan hasil klasifikasi
        print(f'Kategori Kopi: {predicted_category}')
    except json.JSONDecodeError:
        print("Gagal menguraikan data JSON")

    # Beri sedikit jeda sebelum membaca data berikutnya
    time.sleep(1)
