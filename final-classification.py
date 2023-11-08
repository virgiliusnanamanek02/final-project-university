import serial
import time
import tensorflow as tf
import numpy as np

# Buat koneksi dengan Arduino
ser = serial.Serial('/dev/ttyUSB0', 9600) 

# Muat model yang sudah dilatih
model = tf.keras.models.load_model('./data/model-jst')

while True:
    # Baca data dari Arduino
    data = ser.readline().decode('utf-8').strip()
    
    # Praproses data (misalnya, pisahkan nilai suhu, kelembaban, mq7Ppm, mq135Ppm, mq136Ppm)
    data_list = data.split(',')
    suhu = float(data_list[0])
    kelembaban = float(data_list[1])
    mq7Ppm = float(data_list[2])
    mq135Ppm = float(data_list[3])
    mq136Ppm = float(data_list[4])
    
    # Buat data input sesuai dengan model Anda
    input_data = np.array([[mq7Ppm, mq135Ppm, mq136Ppm, suhu, kelembaban]])
    
    # Klasifikasikan data dengan model
    predictions = model.predict(input_data)
    
    # Temukan indeks kategori dengan probabilitas tertinggi
    predicted_category = np.argmax(predictions, axis=1)
    
    # Tampilkan hasil klasifikasi
    print(f'Kategori Kopi: {predicted_category}')

    # Beri sedikit jeda sebelum membaca data berikutnya
    time.sleep(1)
