import numpy as np
import pandas as pd

data = pd.read_csv('data_latih.csv')
X = data[['CO (MQ-7) PPM', 'NH3 (MQ-135) PPM', 'H2S (MQ-136) PPM']].values
y = data[['Kopi Sidikalang', 'Kopi Toraja', 'Kopi Temanggung', 'Kopi Manggarai']].values

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

input_neurons = X.shape[1]
hidden_neurons = 4
output_neurons = y.shape[1]

np.random.seed(0)

weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_neurons))

weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))


epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):
    # Forward Propagation
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_input)

    # Backpropagation
    error = y - predicted_output
    output_gradient = predicted_output * (1 - predicted_output) * error

    hidden_error = np.dot(output_gradient, weights_hidden_output.T)
    hidden_gradient = hidden_output * (1 - hidden_output) * hidden_error

    weights_hidden_output += np.dot(hidden_output.T, output_gradient) * learning_rate
    bias_output += np.sum(output_gradient, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += np.dot(X.T, hidden_gradient) * learning_rate
    bias_hidden += np.sum(hidden_gradient, axis=0, keepdims=True) * learning_rate

def predict(X):
    hidden_output = sigmoid(np.dot(X, weights_input_hidden) + bias_hidden)
    predicted_output = sigmoid(np.dot(hidden_output, weights_hidden_output) + bias_output)
    return predicted_output

data_test = np.array([[10, 20, 30]])  # Ganti dengan nilai ppm dari masing-masing sensor gas
hasil_prediksi = predict(data_test)

# Interprestasi hasil_prediksi menjadi jenis kopi
threshold = 0.5
jenis_kopi = []
for prediksi in hasil_prediksi[0]:
    if prediksi > threshold:
        jenis_kopi.append(1)
    else:
        jenis_kopi.append(0)

# Daftar jenis kopi
daftar_jenis_kopi = ['Kopi Sidikalang', 'Kopi Toraja', 'Kopi Temanggung', 'Kopi Manggarai']

# Menampilkan jenis kopi yang diprediksi
print("Hasil Prediksi (Kombinasi Biner):")
print(jenis_kopi)

jenis_kopi_final = [daftar_jenis_kopi[i] for i in range(len(jenis_kopi)) if jenis_kopi[i] == 1]

print("Jenis Kopi:")
print(jenis_kopi_final)