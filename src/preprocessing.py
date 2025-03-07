import pandas as pd

# Charger les fichiers CSV
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Afficher les premières lignes pour inspection
print(train_data.head())


from sklearn.preprocessing import StandardScaler

# Normaliser les données si nécessaire
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)

# Identifier les indices des valeurs manquantes dans le test
missing_indices = test_data[test_data.isnull()].index
