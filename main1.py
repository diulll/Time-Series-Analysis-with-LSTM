import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# Updated: 2026-01-09

diabetes = load_diabetes()

# Gunakan target sebagai runtun waktu
data = diabetes.target

#Ambil 40 Data Pertama (Agar Sederhana)
data = data[:40]

#Visualisasi data
plt.plot(data, marker='o')
plt.title("Dataset Diabetes (40 Observasi Pertama)")
plt.xlabel("Waktu")
plt.ylabel("Nilai Target")
plt.grid()
plt.show()
scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = scaler.fit_transform(data.reshape(-1,1))
#Gunakan window = 3
def create_dataset(data, window=3):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window, 0])
        y.append(data[i+window, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(data_scaled, window=3)

X = X.reshape((X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(Input(shape=(3,1)))
model.add(LSTM(50, activation='tanh'))
model.add(Dense(1))

model.compile(
    optimizer='adam',    loss='mean_squared_error'
)

model.summary()
model.fit(
    X, y,
    epochs=100,
    batch_size=1,
    verbose=1
)
y_pred = model(X, training=False)

y_pred_inv = scaler.inverse_transform(y_pred.numpy())
y_actual_inv = scaler.inverse_transform(y.reshape(-1,1))

#Visualisasi prediksi
plt.plot(y_actual_inv, label='Data Aktual')
plt.plot(y_pred_inv, label='Prediksi LSTM')
plt.title("Prediksi LSTM – Dataset Diabetes")
plt.xlabel("Waktu")
plt.ylabel("Nilai")
plt.legend()
plt.grid()
plt.show()
last_window = data_scaled[-3:]
last_window = last_window.reshape((1,3,1))

forecast = model(last_window, training=False)
forecast_inv = scaler.inverse_transform(forecast.numpy())

print("Prediksi periode berikutnya:", forecast_inv[0][0])

