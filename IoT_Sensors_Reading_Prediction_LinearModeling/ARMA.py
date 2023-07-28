import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Chargement des données
data = pd.read_csv('telva_dataset.csv', delimiter=';', decimal = '.', parse_dates=[0], date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y', errors='coerce'))
# Il est important que les dates soient bien formatées pour que la méthode ARMA fonctionne correctement.

data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

water_flow = data.set_index('date')['water_flow'].fillna(data['water_flow'].mean())

# Différence des données
diff = water_flow.diff().dropna()

# Séparation entre les données de test et les données d'entrainement
train_size_ARMA = int(len(water_flow) * 0.8)
test_data_ARMA , train_data_ARMA = water_flow[train_size_ARMA:], water_flow[:train_size_ARMA]

# Modèle ARMA
model = ARIMA(train_data_ARMA, order=(4,0, 1))
model_fit = model.fit()

# Graphique des résidus 
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
anomaly_dates = train_data_ARMA.index[abs(model.fit().resid) > anomaly_threshold]
print("Dates des anomalies détectées :")
print(anomaly_dates)

# Graphiques montrant les anmomalies
plt.plot(model.fit().resid, label='Résidus')
plt.plot(anomaly_dates, anomalies, 'ro', label='Anomalies')
plt.axhline(y=anomaly_threshold, color='r', linestyle='--', label='Limite supérieure')
plt.legend()
plt.title("Anomalies détectées")
plt.show()

# qualité du modèle ARIMA 
aic = model_fit.aic
bic = model_fit.bic

# Afficher les résultats
print("AIC:", aic)
print("BIC:", bic)

# prédiction avec les données train_data et test_data
y_pred = model_fit.predict(start=len(train_data_ARMA), end=len(train_data_ARMA) + len(test_data_ARMA) - 1)
y_true = test_data_ARMA

common_period = pd.date_range(start=test_data_ARMA.index.min(), end='2017-12-31').intersection(y_pred.index)
idx_pred = [y_pred.index.get_loc(date) for date in common_period]
idx_true = [y_true.index.get_loc(date) for date in common_period]

y_pred_common = y_pred.iloc[idx_pred]
y_true_common = y_true.iloc[idx_true]

# Graphique avec les prédictions
dta = water_flow
dta.index = pd.date_range(start='1990-01-01', end='2017-12-31', freq='D')
fig, ax = plt.subplots()
ax = dta.loc['1990-01-01':].plot(ax=ax)
plot_predict(model_fit, test_data_ARMA.index.min(), '2017-12-31', ax=ax)
plt.show()


# calcul des métrics
mse = mean_squared_error(y_true_common , y_pred_common)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true_common, y_pred_common)
r2 = r2_score(y_true_common , y_pred_common)
mape = mean_absolute_percentage_error(y_true_common , y_pred_common)

import pprint
pprint.pprint({'mape': mape,
               'mae': mae,
               'mse': mse,
               'rmse': rmse,
               'R²': r2})

print('------------------------------------------------------')
df = pd.concat([y_pred_common, test_data_ARMA], axis=1)
df.columns = ['y_pred_common', 'test_data']  # Renommez les colonnes selon vos besoins

# Affichage du DataFrame
print(df)






















