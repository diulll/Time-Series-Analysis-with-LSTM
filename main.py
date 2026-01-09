import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# Updated: 2026-01-09

data = np.array([
120,128,136,140,145,150,158,162,170,175,
180,185,190,194,200,205,210,215,220,225,
230,235,240,242,250,255,260,265,270,275,
280,285,290,294,300,305,310,315,320,325
])

plt.plot(data, marker='o')
plt.title("Data Penjualan (40 Periode)")
plt.xlabel("Waktu")
plt.ylabel("Nilai")
plt.grid()
plt.show()
scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = scaler.fit_transform(data.reshape(-1,1))
#Gunakan window = 3
#Artinya: 3 periode sebelumnya → prediksi 1 periode berikutnya

def create_dataset(data, window_size=3):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, 0])
        y.append(data[i+window_size, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(data_scaled, window_size=3)
X = X.reshape((X.shape[0], X.shape[1], 1))
model = Sequential()
model.add(Input(shape=(3, 1)))
model.add(LSTM(50, activation='tanh'))
model.add(Dense(1))
model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

model.summary()
history = model.fit(
    X, y,
    epochs=100,
    batch_size=1,
    verbose=1
)
y_pred = model(X, training=False)

forecast = model(last_window, training=False)

# Inverse scaling
y_pred_inv = scaler.inverse_transform(y_pred.numpy())
forecast_inv = scaler.inverse_transform(forecast.numpy())


#Visualisasi prediksi
plt.plot(y_actual_inv, label="Data Aktual")
plt.plot(y_pred_inv, label="Prediksi LSTM")
plt.title("Hasil Prediksi LSTM")
plt.xlabel("Waktu")
plt.ylabel("Nilai")
plt.legend()
plt.grid()
plt.show()
last_window = data_scaled[-3:]
last_window = last_window.reshape((1,3,1))
forecast = model.predict(last_window)
forecast_inv = scaler.inverse_transform(forecast)

print("Prediksi periode ke-41:", forecast_inv[0][0])

