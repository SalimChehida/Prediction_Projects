import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, GlobalAveragePooling1D, RepeatVector
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Chargement des données
data = pd.read_csv('telva_dataset.csv', delimiter=';', decimal='.', parse_dates=[0], date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y', errors='coerce'))
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
water_flow = data.set_index('date')['water_flow'].fillna(data['water_flow'].mean())
dates = data['date']

n_features = 1

def prepare_data_cnn_lstm(data, lookback, horizon, n_features):
    X = []
    y = []
    for i in range(lookback, len(data)-horizon+1):
        X.append(data[i-lookback:i])
        y.append(data[i:i+horizon, -1])
    X = np.array(X)
    X = np.expand_dims(X, axis=2)
    return X, np.array(y)

lookback = 24*6 # deux semaines
horizon = 7 # une semaine

water_flow_val = water_flow.values.reshape(-1, 1)
X, y = prepare_data_cnn_lstm(water_flow_val, lookback, horizon, n_features=n_features)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(lookback, n_features)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
model.add(GlobalAveragePooling1D())  # Réduction des dimensions
model.add(RepeatVector(horizon))  # Répétition des sorties
model.add(LSTM(50, return_sequences=True))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=2)

# Prédiction sur les données de test
y_pred = model.predict(X_test)

# Dates correspondant aux données de test
dates_test = dates[train_size + lookback + horizon - 1:]

y_test_ind = pd.Series(y_test[:, 0], index=dates[len(water_flow) - len(y_test):].tolist())
y_pred_ind = pd.Series(y_pred[:, 0].flatten(), index=dates_test.tolist())

# Affichage de la série brute et des prédictions
plt.figure(figsize=(12, 6))
plt.plot(water_flow, label='Origine')
#plt.plot(y_test_ind, label='Série brute')
plt.plot(y_pred_ind, label='Prédictions')
plt.xlabel('Date')
plt.ylabel("Débit d'eau")
plt.title("Débit d'eau - Série brute vs Prédictions")
plt.legend()
plt.grid(True)
plt.show()

y_test_2d = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))
y_pred_2d = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1]))


mse = mean_squared_error(y_test_2d, y_pred_2d)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_2d, y_pred_2d)
r2 = r2_score(y_test_2d, y_pred_2d)
mape = mean_absolute_percentage_error(y_test_2d, y_pred_2d)

import pprint
pprint.pprint({'mape': mape,
               'mae': mae,
               'mse': mse,
               'rmse': rmse,
               'R²': r2})

# Affichage des séries dans un Data Frame

print('------------------------------------------------------')

df = pd.concat([y_test_ind, y_pred_ind], axis=1)
df.columns = ['y_test', 'y_pred']  

# Affichage du DataFrame
print(df)

# Calcul d'anomalies 
# Calcul des erreurs de prédiction
errors = np.abs(y_pred_ind - y_test_ind)

# Calcul des statistiques des erreurs
mean_error = np.mean(errors)
std_error = np.std(errors)

# Définition du seuil d'anomalie (par exemple, en utilisant la moyenne + 3 * écart-type)
threshold = mean_error + 3 * std_error

# Identification des anomalies
anomalies = errors > threshold

# Nombre d'anomalies détectées
num_anomalies = sum(anomalies)

# Liste des anomalies
anomaly_dates = y_test_ind[anomalies].index.tolist()
print(anomaly_dates)

# Affichage du nombre d'anomalies
print("Nombre d'anomalies détectées :", num_anomalies)

# Affichage des prédictions, des valeurs réelles et des anomalies
plt.plot(water_flow, label='Origine')
plt.plot(y_test_ind, label='Données réelles')
plt.plot(y_pred_ind, label='Prédictions')
plt.scatter(y_test_ind[anomalies].index, y_test_ind[anomalies], color='red', label='Anomalies')
plt.xlabel('Temps')
plt.ylabel("Débit d'eau")
plt.legend()
plt.show()

# Export en csv
df.to_csv('data_CRNN.csv', index=True)
