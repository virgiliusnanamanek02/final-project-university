import os
import numpy as np
import pandas as pd
import PySimpleGUI as sg
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adagrad
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from itertools import product

save_path = "./data/model-jst-02"
training_path = "./data/DataLatihKopi.csv"
testing_path = "./data/Manggarai-Train.csv"

# Fungsi untuk melatih model dengan dropout dan optimasi hiperparameter
def train_model(file_path, save_path, hidden_neurons, learning_rate, dropout_rate, X_train, y_train, X_test, y_test):
    # Load dataset
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()
    X = data[['CO (MQ-7) PPM', 'NH3 (MQ-135) PPM', 'H2S (MQ-136) PPM']].values
    y = data[['Kopi Sidikalang', 'Kopi Toraja', 'Kopi Temanggung', 'Kopi Manggarai']].values

    input_neurons = X.shape[1]
    output_neurons = y.shape[1]

    np.random.seed(0)

    model = Sequential()
    model.add(Dense(hidden_neurons, input_dim=input_neurons, activation='relu'))
    model.add(Dropout(dropout_rate))  # Dropout layer with a dropout rate
    model.add(Dense(output_neurons, activation='softmax'))

    epochs = 500

    optimizer = Adagrad(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=epochs, batch_size=10, verbose=0)

    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    
    sg.popup('Pelatihan sudah selesai!')
    sg.popup(f'Accuracy: {accuracy:.2f}')

    if save_path:
        model.save(save_path)  # Simpan model ke lokasi yang telah ditentukan
        sg.popup('Model telah disimpan di: ' + save_path)

    return model

# Load dataset for training and testing
data_train = pd.read_csv(training_path)
data_test = pd.read_csv(testing_path)

X_train = data_train[['CO (MQ-7) PPM', 'NH3 (MQ-135) PPM', 'H2S (MQ-136) PPM']].values
y_train = data_train[['Kopi Sidikalang', 'Kopi Toraja', 'Kopi Temanggung', 'Kopi Manggarai']].values

X_test = data_test[['CO (MQ-7) PPM', 'NH3 (MQ-135) PPM', 'H2S (MQ-136) PPM']].values
y_test = data_test[['Kopi Sidikalang', 'Kopi Toraja', 'Kopi Temanggung', 'Kopi Manggarai']].values

# Optimasi hiperparameter dengan Random Search
hidden_neurons_range = [16, 32, 64]
learning_rate_range = [0.01, 0.1, 0.2]
dropout_rate_range = [0.2, 0.4, 0.6]

best_accuracy = 0
best_model = None
best_hidden_neurons = 0
best_learning_rate = 0
best_dropout_rate = 0

param_combinations = list(product(hidden_neurons_range, learning_rate_range, dropout_rate_range))

for hidden_neurons, learning_rate, dropout_rate in param_combinations:
    model = train_model(training_path, save_path, hidden_neurons, learning_rate, dropout_rate, X_train, y_train, X_test, y_test)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_hidden_neurons = hidden_neurons
        best_learning_rate = learning_rate
        best_dropout_rate = dropout_rate

sg.popup(f'Best Accuracy: {best_accuracy:.2f}')
sg.popup(f'Best Hidden Neurons: {best_hidden_neurons}')
sg.popup(f'Best Learning Rate: {best_learning_rate}')
sg.popup(f'Best Dropout Rate: {best_dropout_rate}')
