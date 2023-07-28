import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Charger les données
data = pd.read_csv('telva_dataset.csv', delimiter=';', decimal='.', parse_dates=[0], date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y', errors='coerce'))
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
water_flow = data.set_index('date')['water_flow'].fillna(data['water_flow'].mean())

# Séparer les données en ensembles d'entraînement et de test
train_data, test_data = train_test_split(water_flow, test_size=0.2, shuffle=False, random_state=42)

# Détection d'anomalies avec Isolation Forest
outlier_detector = IsolationForest(contamination=0.1)
outlier_detector.fit(train_data.to_frame())

# Prédiction avec régression linéaire
regressor = LinearRegression()
regressor.fit(train_data.values.reshape(-1, 1), train_data.values.reshape(-1, 1))

# Détection d'anomalies sur les données de test
outliers = outlier_detector.predict(test_data.values.reshape(-1, 1))

# Prédictions sur les données de test
predictions = regressor.predict(test_data.values.reshape(-1, 1))

# Afficher les résultats
plt.figure(figsize=(10, 6))
plt.plot(water_flow.index, water_flow, label='Série brute')
plt.plot(test_data.index, predictions, label='Prédictions')
plt.xlabel('Date')
plt.ylabel('Water Flow')
plt.title('Prédictions de la série Water Flow')
plt.legend()
plt.show()

# calcul des métrics
mse = mean_squared_error(test_data , predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_data, predictions)
r2 = r2_score(test_data , predictions)
mape = mean_absolute_percentage_error(test_data , predictions)

import pprint
pprint.pprint({'mape': mape,
               'mae': mae,
               'mse': mse,
               'rmse': rmse,
               'R²': r2})
