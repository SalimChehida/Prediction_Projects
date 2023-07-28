import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess


data = pd.read_csv('telva_dataset.csv', delimiter=';', decimal = '.', parse_dates=[0], date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y', errors='coerce'))
# Il est important que les dates soient bien formatées pour que la méthode ARIMA fonctionne correctement.

data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

water_flow = data.set_index('date')['water_flow'].fillna(data['water_flow'].mean())

# Plot 1: AR parameter = +0.9
plt.subplot(2, 1, 1)
ar1 = np.array([1, -0.9])
ma1 = np.array([1])
AR_object1 = ArmaProcess(ar1, ma1)
plt.plot(water_flow);


from statsmodels.graphics.tsaplots import plot_acf


# Plot 1: AR parameter = +0.9
plot_acf(water_flow, alpha=0.05, lags=100);

import statsmodels.api as sm

# Definir le modèle
model = sm.tsa.SARIMAX(water_flow, order=(1, 1, 4))

# ajuster le modèle
results = model.fit()

# résumé des info du modèle
print(results.summary())

# Print out the estimate for the constant and for phi
print("When the true phi=0.9, the estimate of phi (and the constant) are:")
print(results.params)


mod = sm.tsa.SARIMAX(water_flow, order=(1, 0,1))
res = mod.fit()

forecast = res.forecast(steps=12)

# Visualiser les prévisions
res.plot_diagnostics()
plt.show()

# prédictions pour ce modele
preds = res.predict(start='2015-01-01', end='2017-12-31')

# visualisations
plt.plot(water_flow)
plt.plot(preds)
plt.legend(['water_flow', 'forecast'])
plt.show()


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Calcul du Mean Squared Error (MSE)
mse = mean_squared_error(water_flow['2015-01-01':'2017-12-31'], preds)
print("MSE :", mse)

# Calcul du Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print("RMSE :", rmse)

# Calcul du Mean Absolute Error (MAE)
mae = mean_absolute_error(water_flow['2015-01-01':'2017-12-31'], preds)
print("MAE :", mae)

# Calcul du coefficient de détermination (R-squared)
r2 = r2_score(water_flow['2015-01-01':'2017-12-31'], preds)
print("R-squared :", r2)

























