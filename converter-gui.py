import serial
import csv
import json
import PySimpleGUI as sg

# Inisialisasi koneksi Serial dengan Arduino
ser = serial.Serial('COM3', 9600)  # Ganti 'COM3' dengan port Serial yang sesuai
ser.flushInput()

# Tampilan layout GUI untuk pengaturan nama file
file_layout = [
    [sg.Text('Masukkan Nama File CSV:'), sg.Input(key='file_name')],
    [sg.Button('OK')]
]

# Membuat GUI window untuk pengaturan nama file
file_window = sg.Window('Nama File CSV', file_layout)

# Looping untuk event handling pada window pengaturan nama file
while True:
    event, values = file_window.read()
    if event == sg.WINDOW_CLOSED:
        exit()
    elif event == 'OK':
         file_name = values['file_name'] + '.csv'
         break

# Menutup window pengaturan nama file
file_window.close()

# Buat file CSV untuk menyimpan dataset
with open(file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Suhu (C)', 'Kelembaban (%)', 'CO (MQ-7) PPM', 'NH3 (MQ-135) PPM', 'H2S (MQ-136) PPM'])

    try:
        data_count = 0
        while data_count < 30:
            # Baca data dari Serial Monitor
            data = ser.readline().decode('utf-8').rstrip()
            print(data)  # Tampilkan data di terminal (opsional)

            # Ubah data JSON menjadi dictionary
            data_dict = json.loads(data)

            # Ambil nilai suhu, kelembaban, dan emisi gas dari dictionary
            temperature = data_dict['suhu']
            humidity = data_dict['kelembaban']
            mq7Ppm = data_dict['mq7Ppm']
            mq135Ppm = data_dict['mq135Ppm']
            mq136Ppm = data_dict['mq136Ppm']

            # Tulis data ke dalam file CSV
            writer.writerow([temperature, humidity, mq7Ppm, mq135Ppm, mq136Ppm])

            data_count += 1
            
    except KeyboardInterrupt:
        print("Pengambilan data selesai.")
        sg.popup('Data telah berhasil diambil!')
