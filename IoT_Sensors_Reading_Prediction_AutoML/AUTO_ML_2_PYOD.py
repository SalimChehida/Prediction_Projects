import pandas as pd
from pyod.models.auto_encoder import AutoEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np

# Charger les données de séries chronologiques
data = pd.read_csv('telva_dataset.csv', delimiter=';', decimal='.', parse_dates=[0], date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y', errors='coerce'))
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
water_flow = data.set_index('date')['water_flow'].fillna(data['water_flow'].mean())
dates = data['date']

# Entraîner un modèle d'auto-encodeur
model = AutoEncoder(hidden_neurons=[1, 1, 1, 1])

# Entraîner le modèle sur les données
model.fit(water_flow.values.reshape(-1, 1))

# Faire des prédictions sur les données
predictions = model.predict(water_flow.values.reshape(-1, 1))

# Obtenir les scores d'anomalie
anomaly_scores = model.decision_scores_

# Tracer les prédictions et les scores d'anomalie
plt.figure(figsize=(12, 6))
plt.plot(data['date'], water_flow, label='Valeurs réelles')
plt.plot(data['date'], predictions, label='Prédictions')
plt.scatter(data['date'], water_flow, c=anomaly_scores, cmap='RdYlGn_r', label="Scores d'anomalie")
plt.colorbar(label="Score d'anomalie")
plt.xlabel('date')
plt.ylabel('Water Flow')
plt.legend()
plt.show()

# calcul des métrics
mse = mean_squared_error(water_flow , predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(water_flow, predictions)
r2 = r2_score(water_flow , predictions)
mape = mean_absolute_percentage_error(water_flow , predictions)

import pprint
pprint.pprint({'mape': mape,
               'mae': mae,
               'mse': mse,
               'rmse': rmse,
               'R²': r2})
