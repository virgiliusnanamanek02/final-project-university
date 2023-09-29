import serial
import csv
import json
import PySimpleGUI as sg

# Inisialisasi koneksi Serial dengan Arduino
ser = serial.Serial('COM3', 9600)  # Ganti 'COM3' dengan port Serial yang sesuai
ser.flushInput()

# Fungsi untuk mengambil data dari Arduino dan menyimpannya ke dalam file CSV
def collect_data(file_name, data_type, coffee_type):
    # Buat file CSV untuk menyimpan dataset
    with open(file_name, mode='w', newline='') as file:
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

                # Ambil nilai emisi gas dari dictionary
                mq7Ppm = data_dict['mq7Ppm']
                mq135Ppm = data_dict['mq135Ppm']
                mq136Ppm = data_dict['mq136Ppm']

                # Inisialisasi dictionary data untuk ditulis ke CSV
                data_row = {
                    'CO (MQ-7) PPM': mq7Ppm,
                    'NH3 (MQ-135) PPM': mq135Ppm,
                    'H2S (MQ-136) PPM': mq136Ppm,
                    'Kopi Sidikalang': 0,
                    'Kopi Toraja': 0,
                    'Kopi Temanggung': 0,
                    'Kopi Manggarai': 0
                }

                if data_type == 'Pelatihan':
                    # Set nilai sesuai dengan jenis kopi yang dipilih
                    data_row[coffee_type] = 1

                # Tulis data ke dalam file CSV
                writer.writerow(data_row)

                data_count += 1
                
        except KeyboardInterrupt:
            print("Pengambilan data selesai.")
            sg.popup('Data telah berhasil diambil!')

# Tampilan layout GUI
layout = [
    [sg.Text('Jenis Data:'), sg.DropDown(['Pelatihan', 'Pengujian'], key='data_type')],
    [sg.Text('Jenis Kopi:'), sg.DropDown(['Kopi Sidikalang', 'Kopi Toraja', 'Kopi Temanggung', 'Kopi Manggarai'], key='coffee_type')],
    [sg.Text('Nama file:'), sg.InputText(key='file_name')],
    [sg.Button('OK')]
]

# Membuat GUI window
window = sg.Window('Input Data', layout)

# Looping untuk event handling pada window
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    elif event == 'OK':
        data_type = values['data_type']
        coffee_type = values['coffee_type']
        file_name = values['file_name'] + '.csv'

        # Validasi nama file
        if not file_name:
            sg.popup_error('Nama file tidak boleh kosong!')
        else:
            collect_data(file_name, data_type, coffee_type)

# Menutup window
window.close()

