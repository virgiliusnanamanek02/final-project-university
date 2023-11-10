import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import PySimpleGUI as sg

save_path = "./data/model-jst-01"
training_path = "./data/DataLatihKopi.csv"

# Fungsi untuk membuat model
def create_model(learning_rate=0.01, hidden_neurons=16, dropout_rate=0.2):
    input_neurons = 3
    output_neurons = 4

    model = Sequential()
    model.add(Dense(hidden_neurons, input_dim=input_neurons, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(output_neurons, activation='softmax'))

    adagrad = Adagrad(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adagrad, metrics=['accuracy'])

    return model

# Fungsi untuk melatih model
def train_model(file_path, save_path):
    # Load data and preprocess as before
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()
    X = data[['CO (MQ-7) PPM', 'NH3 (MQ-135) PPM', 'H2S (MQ-136) PPM']].values
    y = data[['Kopi Sidikalang', 'Kopi Toraja', 'Kopi Temanggung', 'Kopi Manggarai']].values

    # Normalize the numerical features
    X[:, :3] = (X[:, :3] - X[:, :3].min()) / (X[:, :3].max() - X[:, :3].min())

    input_neurons = X.shape[1]
    output_neurons = y.shape[1]

    np.random.seed(0)

    model = KerasClassifier(build_fn=create_model, verbose=0)

    # Define the grid of parameters to search
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'hidden_neurons': [16, 32, 64],
        'dropout_rate': [0.1, 0.2, 0.3],
        'epochs': [200, 300, 400]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X, y)

    # Get the best model from the search
    best_model = grid_search.best_estimator_

    best_model.model.fit(X, y)

    sg.popup('Pelatihan sudah selesai!')

    if save_path:
        best_model.model.save_weights(save_path)
        sg.popup('Model telah disimpan di: ' + save_path)

    return best_model

# Train the model with parameter optimization
model = train_model(training_path, save_path)
print(model)
