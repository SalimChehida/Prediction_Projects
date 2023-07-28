import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Importation des données
data = pd.read_csv('telva_dataset.csv', delimiter=';', decimal='.', parse_dates=[0], date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y', errors='coerce'))
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
water_flow = data.set_index('date')['water_flow'].fillna(data['water_flow'].mean())
dates = data['date']

# Préparation des données
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[['water_flow', 'water_precipitation', 'water_height', 'water_volume']].values)

lookback = 24 * 7 # 1 an
horizon = 7  # une semaine

X = []
y = []
for i in range(lookback, len(data_scaled) - horizon + 1):
    X.append(data_scaled[i - lookback:i])
    y.append(data_scaled[i:i + horizon, 0])  # Utilisation uniquement de la série "water_flow"
X = np.array(X)
y = np.array(y)

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Construction du modèle GRU avec les autres variables
model = Sequential()
model.add(GRU(64, input_shape=(lookback, 4)))  # 4 variables (water_flow, water_precipitation, water_height, water_volume)
model.add(Dense(horizon))

model.compile(optimizer=Adam(), loss='mse')

# Entraînement du modèle
model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=2)

# Prédiction sur les données de test
y_pred = model.predict(X_test)

# Dates correspondant aux données de test
dates_test = dates[train_size + lookback + horizon - 1:]

# Reconstruction des valeurs d'origine
y_test = y_test * scaler.data_range_[0] + scaler.data_min_[0]
y_pred = y_pred * scaler.data_range_[0] + scaler.data_min_[0]

# Remise en forme des index
y_pred_ind = pd.Series(y_pred[:, 0].flatten(), index=dates_test.tolist())
y_test_ind = pd.Series(y_test[:, 0].flatten(), index=dates_test.tolist())

# Affichage de la série brute de test et des prédictions
plt.figure(figsize=(12, 6))
plt.plot(water_flow.loc[dates_test], label='Série brute')
plt.plot(y_pred_ind, label='Prédictions')
plt.xlabel('Date')
plt.ylabel("Débit d'eau")
plt.title("Débit d'eau - Série brute de test vs Prédictions")
plt.legend()
plt.grid(True)
plt.show()

# Evaluation des performances
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("Performance du modèle GRU - multi:")
print("MSE :", mse)
print("RMSE :", rmse)
print("MAE :", mae)
print("R2 :", r2)
print("MAPE :", mape)

# Affichage des séries dans un Data Frame

print('------------------------------------------------------')

df = pd.concat([y_test_ind, y_pred_ind], axis=1)
df.columns = ['y_test', 'y_pred']  # Renommez les colonnes selon vos besoins

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
df.to_csv('E:/Deuxième année/Stage/stage_verimag/data_GRU_multi.csv', index=True)