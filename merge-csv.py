import PySimpleGUI as sg
import pandas as pd

def merge_csv_files(file_paths, output_file):
    # Baca file pertama untuk mendapatkan header
    df = pd.read_csv(file_paths[0])
    
    # Baca file-file lainnya dan gabungkan dengan header yang sama
    for file_path in file_paths[1:]:
        df = pd.concat([df, pd.read_csv(file_path)], ignore_index=True)
    
    # Simpan dataframe ke file CSV tujuan
    df.to_csv(output_file, index=False)

sg.theme('DarkGrey5')

layout = [
    [sg.Text('Pilih file CSV untuk digabungkan:')],
    [sg.Input(key='file1'), sg.FileBrowse(file_types=(('CSV Files', '*.csv'),))],
    [sg.Input(key='file2'), sg.FileBrowse(file_types=(('CSV Files', '*.csv'),))],
    [sg.Input(key='file3'), sg.FileBrowse(file_types=(('CSV Files', '*.csv'),))],
    [sg.Input(key='file4'), sg.FileBrowse(file_types=(('CSV Files', '*.csv'),))],
    [sg.Button('Merge')],
    [sg.Text('', size=(30, 1), key='message')],
]

window = sg.Window('CSV Merger', layout)

while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    if event == 'Merge':
        file1 = values['file1']
        file2 = values['file2']
        file3 = values['file3']
        file4 = values['file4']
        
        if file1 and file2 and file3 and file4:
            file_paths = [file1, file2, file3, file4]
            output_file = 'merged.csv'
            merge_csv_files(file_paths, output_file)
            window['message'].update(f'Semua file telah digabungkan menjadi {output_file}')
        else:
            window['message'].update('Silakan pilih semua file terlebih dahulu.')

window.close()
