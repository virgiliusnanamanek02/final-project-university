import serial
import csv
import json
import PySimpleGUI as sg

# Inisialisasi koneksi Serial dengan Arduino
# Ganti 'COM3' dengan port Serial yang sesuai
ser = serial.Serial('COM3', 9600)
ser.flushInput()

# Tampilan layout GUI untuk input data
data_layout = [
    [sg.Text('Jenis Data:'), sg.DropDown(
        ('Pelatihan', 'Pengujian'), key='data_type')],
    [sg.Text('Jenis Kopi:'), sg.DropDown(('Kopi Sidikalang', 'Kopi Toraja',
                                          'Kopi Temanggung', 'Kopi Manggarai'), key='coffee_type')],
    [sg.Text('Nama File CSV:'), sg.Input(key='file_name')],
    [sg.Button('OK')]
]

# Membuat GUI window untuk input data
data_window = sg.Window('Input Data', data_layout)

# Looping untuk event handling pada window input data
while True:
    event, values = data_window.read()
    if event == sg.WINDOW_CLOSED:
        exit()
    elif event == 'OK':
        data_type = values['data_type']  # Jenis data yang dipilih
        file_name = values['file_name'] + '.csv'
        break

# Menutup window input data
data_window.close()

# Buat file CSV untuk menyimpan dataset
with open(file_name, mode='w', newline='') as file:
    if data_type == 'Pelatihan':
        # Tambahkan kolom jenis kopi yang sesuai
        fieldnames = ['CO (MQ-7) PPM', 'NH3 (MQ-135) PPM', 'H2S (MQ-136) PPM',
                      'Kopi Sidikalang', 'Kopi Toraja', 'Kopi Temanggung', 'Kopi Manggarai']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        try:
            data_count = 0
            while data_count < 30:
                # Baca data dari Serial Monitor
                data = ser.readline().decode('utf-8').rstrip()
                print(data)  # Tampilkan data di terminal (opsional)

                # Ubah data JSON menjadi dictionary
                data_dict = json.loads(data)

                # Ambil emisi gas dari dictionary
                mq7Ppm = data_dict['mq7Ppm']
                mq135Ppm = data_dict['mq135Ppm']
                mq136Ppm = data_dict['mq136Ppm']

                # Inisialisasi kolom jenis kopi dengan nilai 0
                coffee_values = {'Kopi Sidikalang': 0, 'Kopi Toraja': 0,
                                 'Kopi Temanggung': 0, 'Kopi Manggarai': 0}
                selected_coffee = values['coffee_type']
                coffee_values[selected_coffee] = 1

                # Tulis data ke dalam file CSV
                row_data = {'CO (MQ-7) PPM': mq7Ppm, 'NH3 (MQ-135) PPM': mq135Ppm,
                            'H2S (MQ-136) PPM': mq136Ppm, **coffee_values}
                writer.writerow(row_data)

                data_count += 1

        except KeyboardInterrupt:
            print("Pengambilan data selesai.")
            sg.popup('Data telah berhasil diambil!')

    elif data_type == 'Pengujian':
        # Kolom yang sama seperti data pelatihan
        fieldnames = ['CO (MQ-7) PPM', 'NH3 (MQ-135) PPM', 'H2S (MQ-136) PPM',
                      'Kopi Sidikalang', 'Kopi Toraja', 'Kopi Temanggung', 'Kopi Manggarai']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        try:
            data_count = 0
            while data_count < 30:
                # Baca data dari Serial Monitor
                data = ser.readline().decode('utf-8').rstrip()
                print(data)  # Tampilkan data di terminal (opsional)

                # Ubah data JSON menjadi dictionary
                data_dict = json.loads(data)

                # Ambil emisi gas dari dictionary
                mq7Ppm = data_dict['mq7Ppm']
                mq135Ppm = data_dict['mq135Ppm']
                mq136Ppm = data_dict['mq136Ppm']

                # Inisialisasi kolom jenis kopi dengan nilai kosong
                coffee_values = {'Kopi Sidikalang': '', 'Kopi Toraja': '',
                                 'Kopi Temanggung': '', 'Kopi Manggarai': ''}

                # Tulis data ke dalam file CSV
                row_data = {'CO (MQ-7) PPM': mq7Ppm, 'NH3 (MQ-135) PPM': mq135Ppm,
                            'H2S (MQ-136) PPM': mq136Ppm, **coffee_values}
                writer.writerow(row_data)

                data_count += 1

        except KeyboardInterrupt:
            print("Pengambilan data selesai.")
            sg.popup('Data telah berhasil diambil!')
