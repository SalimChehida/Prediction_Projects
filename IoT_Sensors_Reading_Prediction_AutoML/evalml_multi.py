import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from evalml import AutoMLSearch
import pprint
import evalml
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Importer les données
data = pd.read_csv('telva_dataset.csv', delimiter=';', decimal='.', parse_dates=[0], date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y', errors='coerce'), encoding='utf-8')
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
data['water_volume_variation'] = data['water_volume_variation'].fillna(0)  # Remplacer les valeurs manquantes par zéro

# Créer une dataframe pour EvalML
df = pd.DataFrame({
    'date': data['date'],
    'water_flow': data['water_flow'],
    'water_precipitation': data['water_precipitation'],
    'water_height': data['water_height'],
    'water_volume': data['water_volume'],
    'water_volume_variation': data['water_volume_variation']
})

# Définir la colonne de la cible
target_column = 'water_flow'

# Diviser les données en ensembles d'entraînement et de test
train_data = df[:-1825] 
test_data = df[-1825:]

# Rechercher automatiquement le meilleur modèle avec EvalML
automl = AutoMLSearch(X_train=train_data.drop(columns=target_column),
                      y_train=train_data[target_column],
                      problem_type='regression',
                      objective='mse',
                      additional_objectives=['mae', 'Root Mean Squared Error', 'r2'],
                      allowed_model_families=['random_forest', 'gradient_boosting'])
automl.search()

# Sélectionner le meilleur pipeline trouvé
best_pipeline = automl.best_pipeline

# Entraîner le meilleur pipeline sur toutes les données d'entraînement
best_pipeline.fit(train_data.drop(columns=target_column), train_data[target_column])

# Effectuer des prédictions sur les données de test
predictions = best_pipeline.predict(test_data.drop(columns=target_column))

# Calculer les métriques
mae = mean_absolute_error(test_data[target_column], predictions)
mse = mean_squared_error(test_data[target_column], predictions)
rmse = np.sqrt(mse)
r2 = np.corrcoef(predictions, test_data[target_column])[0,1]

# Afficher les métriques
metrics = {'mae': mae, 'mse': mse, 'rmse': rmse, 'R²': r2}
pprint.pprint(metrics)

# Tracer le graphique des prédictions
plt.figure(figsize=(10, 6))
plt.plot(test_data['date'], test_data[target_column], label='Réel')
plt.plot(test_data['date'], predictions, label='Prédiction')
plt.xlabel('Date')
plt.ylabel("Débit d'eau")
plt.title('Prédictions du débit d\'eau')
plt.legend()
plt.show()

# Fusionner les DataFrames en utilisant l'index comme clé de fusion
df = test_data.merge(predictions, left_index=True, right_index=True, how='inner')

# Afficher le DataFrame fusionné
print(df)

# Créer une série à partir d'une colonne d'un DataFrame
test = pd.Series(test_data['water_flow'])

# Changer l'index de la série en utilisant les valeurs de la colonne 'date' du DataFrame
test_with_new_index = pd.Series(test.values, index=test_data['date'])
predictions_with_new_index = pd.Series(predictions.values, index=test_data['date'])

# Calcul d'anomalies 
# Calcul des erreurs de prédiction
errors = np.abs(predictions_with_new_index - test_with_new_index)

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
anomaly_dates = test_with_new_index[anomalies].index.tolist()
print(anomaly_dates)

# Affichage du nombre d'anomalies
print("Nombre d'anomalies détectées :", num_anomalies)

# Affichage des prédictions, des valeurs réelles et des anomalies
plt.plot(data['date'], data['water_flow'], label='Origine')
plt.plot(test_with_new_index, label='Données réelles')
plt.plot(predictions_with_new_index, label='Prédictions')
plt.scatter(test_with_new_index[anomalies].index, test_with_new_index[anomalies], color='red', label='Anomalies')
plt.xlabel('Temps')
plt.ylabel("Débit d'eau")
plt.legend()
plt.show()
