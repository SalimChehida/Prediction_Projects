import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="No frequency information was provided, so inferred frequency D will be used.", module="statsmodels.tsa.base.tsa_model")

# Import des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import register_matplotlib_converters
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_predict
from pandas.plotting import autocorrelation_plot 
register_matplotlib_converters()

# Chargement des données
data = pd.read_csv('telva_dataset.csv', delimiter=';', decimal = '.', parse_dates=[0], date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y', errors='coerce'))
# Il est important que les dates soient bien formatées pour que la méthode ARIMA fonctionne correctement.

data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

# Correspondance avec la date et remplacement des valeurs manquantes par la moyenne
water_flow = data.set_index('date')['water_flow'].fillna(data['water_flow'].mean())

# Test de stationnarité des données
def test_stationarity(timeseries):

    # Test de Dickey-Fuller
    print('Résultats du test de Dickey-Fuller :')
    test = adfuller(timeseries)
    test_output = pd.Series(test[0:4], index=['Statistique de test', 'p-value', 'Lags utilisés', 'Nombre d\'observations'])
    for key, value in test[4].items():
        if key not in ['1%', '10%']:
            test_output['Valeur critique (%s)'%key] = value
    print(test_output)

test_stationarity(water_flow)

# ici la série est bien stationnaire donc on peut continuer sans faire un ajustement supplémentaire
# faire la décomposition (tendance, saisonnalité, résidus)
ts = data.set_index('date')['water_flow']
decomposition = sm.tsa.seasonal_decompose(ts)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Graphique de la série brute
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(8,8))
ax.plot(ts.index, trend, color = 'red')
ax.set(title='Tendance de water_flow')

# Graphique de la saisonnalité 
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(8,8))
ax.bar(seasonal.index, seasonal, color = 'green')
ax.set(title='Saisonnalité de water_flow')

# Graphique des résidus 
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(8,8))
ax.plot(ts.index, residual)
ax.set(title='Résidus de water_flow')

# Séparation des données en jeu d'entraînement et jeu de test
train_size = int(len(water_flow) * 0.8)
test_data , train_data = water_flow[train_size:], water_flow[:train_size]

# Prétraitement des données : 
missing_values = data.isnull().sum()
print('Nombre de valeurs manquantes par colonne :\n', missing_values)

# Construction du modèle ARIMA
# Pour déterminer les paramètres du modèle ARIMA, nous allons utiliser la fonction autocorrelation_plot
# Plot de l'autocorrélation de la série stationnaire
## acf
fig, ax = plt.subplots(figsize=(10, 4))
autocorrelation_plot(water_flow.diff().dropna(), ax=ax)
ax.set_xlim([1, len(water_flow.diff().dropna())])
ax.set_ylim(-0.10, 0.10) # Set the y-axis limits to exclude the first value
ax.set_title("Autocorrélation de la série water_flow")
plt.show()

# pacf
fig, ax = plt.subplots(figsize=(10, 4))
plot_pacf(water_flow.diff().dropna(), ax=ax, lags=range(1, 1000))
ax.set_ylim([-0.1, 0.3])
plt.show()

# Construction du modèle ARIMA
model = ARIMA(train_data, order=(5, 1, 5))
model_fit = model.fit()

# critères d'évaluation de p, d et q
aic = model_fit.aic
bic = model_fit.bic

# Afficher les résultats
print("AIC:", aic)
print("BIC:", bic)

# Vérification du résumé du modèle pour voir si tout est correct
print(model_fit.summary())

# Affichage des résidus
residuals = pd.DataFrame(model_fit.resid)
residuals.plot(title = 'Résidus du modèle ARIMA')
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())

# Calculer la limite supérieure des résidus pour détecter les anomalies
mean_resid = np.mean(model.fit().resid)
std_resid = np.std(model.fit().resid)
anomaly_threshold = mean_resid + 4 * std_resid

# Détecter les anomalies dans les résidus
anomalies = model.fit().resid[abs(model.fit().resid) > anomaly_threshold]

# Afficher les dates des anomalies détectées
anomaly_dates = train_data.index[abs(model.fit().resid) > anomaly_threshold]
print("Dates des anomalies détectées :")
print(anomaly_dates)

# Graphiques montrant les anmomalies
plt.plot(model.fit().resid, label='Résidus')
plt.plot(anomaly_dates, anomalies, 'ro', label='Anomalies')
plt.axhline(y=anomaly_threshold, color='r', linestyle='--', label='Limite supérieure')
plt.legend()
plt.title("Anomalies détectées")
plt.show()

# prédiction avec les données train_data et test_data
y_pred = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
y_true = test_data

common_period = pd.date_range(start=test_data.index.min(), end='2017-12-31').intersection(y_pred.index)
idx_pred = [y_pred.index.get_loc(date) for date in common_period]
idx_true = [y_true.index.get_loc(date) for date in common_period]

y_pred_common = y_pred.iloc[idx_pred]
y_true_common = y_true.iloc[idx_true]

# Graphique avec les prédictions
dta = water_flow
dta.index = pd.date_range(start='1990-01-01', end='2017-12-31', freq='D')
fig, ax = plt.subplots()
ax = dta.loc['1990-01-01':].plot(ax=ax)
plot_predict(model_fit, test_data.index.min(), '2017-12-31', ax=ax)
plt.show()

# Calcul de la qualité de prédiction
mape = np.mean(np.abs(y_pred_common - y_true_common) / np.abs(y_true_common)) # Mean absolute percentage error
mae = np.mean(np.abs(y_pred_common - y_true_common))
mse = np.mean((y_pred_common - y_true_common) ** 2)
rmse = np.sqrt(np.mean((y_pred_common - y_true_common) ** 2)).round(5)
corr = np.corrcoef(y_pred_common, y_true_common)[0, 1]

mins = np.amin(np.hstack([y_pred_common[:, None], y_true_common[:, None]]), axis=1)
maxs = np.amax(np.hstack([y_pred_common[:, None], y_true_common[:, None]]), axis=1)
minmax = 1 - np.mean(mins / maxs)

import pprint
pprint.pprint({'mape': mape,
               'mae': mae,
               'mse': mse,
               'rmse': rmse,
               'R': corr,
               'minmax': minmax})
print('------------------------------------------------------')
df = pd.concat([y_pred_common, test_data], axis=1)
df.columns = ['y_pred_common', 'test_data']  # Renommez les colonnes selon vos besoins

# Affichage du DataFrame
print(df)


