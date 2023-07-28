import warnings
warnings.filterwarnings("ignore")
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import StandardScaler
from keras.layers import Dropout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Chargement des données
data = pd.read_csv('telva_dataset.csv', delimiter=';', decimal='.', parse_dates=[0], date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y', errors='coerce'))
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')
data['water_volume_variation'] = data['water_volume_variation'].fillna(0)  # Remplacer la valeur manquante par zéro
water_flow = data.set_index('date')['water_flow']
dates = data['date']

# Normalisation des données
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data.drop(['date'], axis=1).values)

# Séparation des données en ensembles d'entraînement et de test
train_size = int(len(normalized_data) * 0.8)
train_data = normalized_data[:train_size]
test_data = normalized_data[train_size:]

# Fonction pour préparer les séquences d'entraînement
def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length, :])
        y.append(data[i+sequence_length, 2])  # Utiliser uniquement la colonne "water_flow" comme variable cible
    return np.array(X), np.array(y)

# Paramètres du modèle RNN
sequence_length = 10  # Longueur des séquences d'entrée
n_features = train_data.shape[1]  # Nombre de caractéristiques (nombre de colonnes dans les données d'entraînement)

# Création des séquences d'entraînement
X_train, y_train = create_sequences(train_data, sequence_length)

# Création du modèle RNN avec couche Dropout
model = Sequential()
model.add(LSTM(64, input_shape=(sequence_length, n_features)))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Préparation des séquences de test
X_test, y_test = create_sequences(test_data, sequence_length)

# Entraînement du modèle
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Prédiction sur les séquences de test
y_pred = model.predict(X_test)

# Inversion de la normalisation des prédictions
y_pred = scaler.inverse_transform(np.concatenate((np.zeros((sequence_length, n_features-1)), y_pred), axis=1))

# Inversion de la normalisation des données réelles
y_test = scaler.inverse_transform(y_test)

y_test_series = pd.Series(y_test[:, 0], index=dates[train_size+sequence_length:train_size+sequence_length+len(y_test)])
y_pred_series = pd.Series(y_pred[:, 0], index=dates[train_size+sequence_length:train_size+sequence_length+len(y_pred)])

# Affichage des prédictions et des données réelles
plt.figure(figsize=(12, 6))
plt.plot(y_test_series, label='Données réelles')
plt.plot(y_pred_series, label='Prédictions')
plt.title('Prédictions de débit d\'eau')
plt.xlabel('Date')
plt.ylabel('Débit d\'eau')
plt.legend()
plt.show()


# Calcul des métriques
mse = mean_squared_error(y_test , y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test , y_pred)
mape = mean_absolute_percentage_error(y_test , y_pred)

print("Performance du modèle LSTM :")
import pprint
pprint.pprint({'mape': mape,
               'mae': mae,
               'mse': mse,
               'rmse': rmse,
               'R²': r2})

print('------------------------------------------------------')
df = pd.concat([y_test_series, y_pred_series], axis=1)
df.columns = ['y_test_series', 'y_pred_series']  # Renommez les colonnes selon vos besoins

# Affichage du DataFrame
print(df)

# Calcul d'anomalies 
# Calcul des erreurs de prédiction
errors = np.abs(y_pred_series - y_test_series)

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
anomaly_dates = y_test_series[anomalies].index.tolist()
print(anomaly_dates)

# Affichage du nombre d'anomalies
print("Nombre d'anomalies détectées :", num_anomalies)

# Affichage des prédictions, des valeurs réelles et des anomalies
plt.plot(water_flow, label='origine')
plt.plot(y_test_series, label='Données réelles')
plt.plot(y_pred_series, label='Prédictions')
plt.scatter(y_test_series[anomalies].index, y_test_series[anomalies], color='red', label='Anomalies')
plt.xlabel('Temps')
plt.ylabel("Débit d'eau")
plt.legend()
plt.show()
