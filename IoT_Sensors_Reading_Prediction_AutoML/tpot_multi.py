from tpot import TPOTRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Charger les données
data = pd.read_csv('telva_dataset.csv', delimiter=';', decimal='.', parse_dates=[0], date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y', errors='coerce'))
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

# Remplacer les valeurs manquantes par 0 pour la série 'water_volume_variation'
data['water_volume_variation'] = data['water_volume_variation'].fillna(0)

# Diviser les données en ensembles d'entraînement et de test manuellement
train_size = int(len(data) * 0.8)  # 80% des données pour l'entraînement
X_train, y_train = data.drop(['date', 'water_flow'], axis=1)[:train_size], data['water_flow'][:train_size]
X_test, y_test = data.drop(['date', 'water_flow'], axis=1)[train_size:], data['water_flow'][train_size:]

# Sélectionner les 20% les plus récents des dates pour les données de test
test_dates = data['date'][train_size:]

# Créer une instance de TPOTRegressor
tpot = TPOTRegressor(generations=10, population_size=50, verbosity=2, random_state=42)

# Lancer l'AutoML
tpot.fit(X_train.values, y_train.values)

# Faire des prédictions sur les données de test
predictions = tpot.predict(X_test.values)

# Réinitialiser l'index des données de test
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Évaluer les performances en utilisant les métriques demandées
mape = mean_absolute_percentage_error(y_test.values, predictions)
mae = mean_absolute_error(y_test.values, predictions)
mse = mean_squared_error(y_test.values, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test.values, predictions)

# Afficher les métriques
metrics = {'mape': mape, 'mae': mae, 'mse': mse, 'rmse': rmse, 'R²': r2}
print(metrics)

# Tracer la série brute et la série de prédiction pour les données de test
plt.figure(figsize=(10, 6))
plt.plot(test_dates, y_test.values, label='Série brute')
plt.plot(test_dates, predictions, label='Série de prédiction', color='r')
plt.xlabel('Date')
plt.ylabel("Débit d'eau")
plt.title('Débit d\'eau - Série brute et Série de prédiction (Données de test)')
plt.legend()
plt.xticks(rotation=45)

plt.show()

print('------------------------------------------------------')

df = pd.DataFrame({'test': y_test.values, 'pred': predictions}, index=test_dates)

print(df)

y_test = pd.Series(y_test.values, index=test_dates)
predictions = pd.Series(predictions, index=test_dates)

# Calcul d'anomalies
# Calcul des erreurs de prédiction
errors = np.abs(predictions - y_test)

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
anomaly_dates = y_test[anomalies].index.tolist()
print(anomaly_dates)

# Affichage du nombre d'anomalies
print("Nombre d'anomalies détectées :", num_anomalies)

# Affichage des prédictions, des valeurs réelles et des anomalies
plt.plot(data['date'], data['water_flow'], label='Origine')
plt.plot(y_test, label='Données réelles')
plt.plot(predictions, label='Prédictions')
plt.scatter(y_test[anomalies].index, y_test[anomalies], color='red', label='Anomalies')
plt.xlabel('Temps')
plt.ylabel("Débit d'eau")
plt.legend()
plt.show()
