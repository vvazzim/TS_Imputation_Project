import pandas as pd

# Charger les fichiers CSV
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Avant l'imputation, identifier les indices des valeurs manquantes dans le jeu de test
missing_indices = test_data[test_data.isnull()].index

# Imputation par interpolation linéaire
test_data_imputed = test_data.interpolate(method='linear')



from statsmodels.tsa.arima.model import ARIMA

# Identifier les indices des valeurs manquantes dans le test
missing_indices = test_data[test_data.isnull()].index


# Appliquer ARIMA pour chaque série du jeu d'entraînement
forecasted_values = []
for i in range(len(train_data)):  # Itérer sur chaque série du train
    model = ARIMA(train_data.iloc[i], order=(5, 1, 0))  # Paramètres à ajuster
    model_fit = model.fit()  # Ajuster le modèle
    forecast = model_fit.forecast(steps=len(test_data.columns))  # Prédire les valeurs manquantes
    forecasted_values.append(forecast)  # Ajouter la prédiction aux résultats

# Remplir les valeurs manquantes dans test_data avec les prévisions d'ARIMA
test_data.iloc[missing_indices] = forecasted_values


# Remplir les valeurs manquantes par les prédictions ARIMA
test_data.iloc[missing_indices] = forecast
