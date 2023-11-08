import PySimpleGUI as sg
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

def train_model_with_cv(file_path, n_folds=5):
    if not file_path:
        sg.popup_error('Pilih file data pelatihan terlebih dahulu!')
        return None

    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() != '.csv':
        sg.popup_error('File harus berekstensi CSV!')
        return None

    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()
    X = data[['CO (MQ-7) PPM', 'NH3 (MQ-135) PPM', 'H2S (MQ-136) PPM']].values
    y = data[['Kopi Sidikalang', 'Kopi Toraja',
              'Kopi Temanggung', 'Kopi Manggarai']].values

    input_neurons = X.shape[1]
    hidden_neurons = 4
    output_neurons = y.shape[1]

    np.random.seed(0)

    model = Sequential()
    model.add(Dense(hidden_neurons, input_dim=input_neurons, activation='relu'))
    model.add(Dropout(0.2))  # Dropout layer with a 20% dropout rate
    model.add(Dense(output_neurons, activation='softmax'))

    epochs = 500
    learning_rate = 0.1

    sgd = SGD(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    
    accuracy_scores = []

    for train_index, val_index in kf.split(X, np.argmax(y, axis=1)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(X_train, y_train, epochs=epochs, batch_size=10, verbose=0)

        predicted_output = model.predict(X_val)
        true_labels = np.argmax(y_val, axis=1)
        accuracy = accuracy_score(true_labels, np.argmax(predicted_output, axis=1))
        accuracy_scores.append(accuracy)

    mean_accuracy = np.mean(accuracy_scores)
    print(f'Mean Cross-Validation Accuracy: {mean_accuracy * 100:.2f}%')
    
    sg.popup(f'Mean Cross-Validation Accuracy: {mean_accuracy * 100:.2f}%')

    return model

layout = [
    [sg.Text('Pilih file data pelatihan (CSV):')],
    [sg.InputText(key='train_file_path', size=(40, 1)), sg.FileBrowse(file_types=(("CSV Files", "*.csv"),))],
    [sg.Button('Pelatihan'), sg.Button('Keluar')],
    [sg.Text('', size=(40, 1), key='result_text')],
]

window = sg.Window('Pelatihan Model', layout)

while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED or event == 'Keluar':
        break

    elif event == 'Pelatihan':
        model = train_model_with_cv(values['train_file_path'])
        if model:
            window['result_text'].update('Pelatihan selesai!')

window.close()
