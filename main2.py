import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# =============================
# 1. Load Dataset
# =============================
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)

# Gunakan satu fitur sebagai "time series"
series = X[['mean radius']].values


# =============================
# 2. Normalisasi
# =============================
scaler = MinMaxScaler(feature_range=(0, 1))
series_scaled = scaler.fit_transform(series)


# =============================
# 3. Sliding Window
# Window Size = 3
# =============================
def create_sliding_window(data, window_size=3):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, 0])
        y.append(data[i+window_size, 0])
    return np.array(X), np.array(y)


window_size = 3
X_sw, y_sw = create_sliding_window(series_scaled, window_size)


# =============================
# 4. Reshaping untuk LSTM
# (samples, time steps, features)
# =============================
X_lstm = X_sw.reshape((X_sw.shape[0], X_sw.shape[1], 1))


# =============================
# 5. Split Data Training
# =============================
train_size = int(0.8 * X_lstm.shape[0])
X_train, y_train = X_lstm[:train_size], y_sw[:train_size]
X_test, y_test = X_lstm[train_size:], y_sw[train_size:]


# =============================
# 6. Model LSTM
# =============================
model = Sequential()
model.add(LSTM(50, activation='tanh', input_shape=(window_size, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# =============================
# 7. Training Model
# Epoch >= 100, Batch Size = 1
# =============================
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=1,
    verbose=1
)


# =============================
# 8. Peramalan 1 Periode ke Depan
# =============================
last_window = series_scaled[-window_size:].reshape((1, window_size, 1))
pred_scaled = model.predict(last_window)
prediction = scaler.inverse_transform(pred_scaled)

print("Prediksi 1 periode ke depan (mean radius):", prediction[0][0])
