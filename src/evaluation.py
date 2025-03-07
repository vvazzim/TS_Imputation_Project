import pandas as pd
import matplotlib.pyplot as plt
import os

# Vérifier si les fichiers existent
print(os.path.exists('data/train.csv'))  # Devrait retourner True si le fichier est accessible
print(os.path.exists('data/test.csv'))   # Devrait retourner True si le fichier est accessible

# Charger les fichiers CSV
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Imputation par interpolation linéaire
test_data_imputed = test_data.interpolate(method='linear')

# Afficher les premières lignes après imputation
print(test_data_imputed.head())

# Identifier les indices des valeurs manquantes dans le test
missing_indices = test_data[test_data.isnull()].index

# Tracer la série avant et après imputation
plt.figure(figsize=(10, 6))
plt.plot(test_data.iloc[0], label='Avant Imputation')
plt.plot(test_data_imputed.iloc[0], label='Après Imputation', linestyle='--')
plt.title('Comparaison des Séries Avant et Après Imputation')
plt.legend()
plt.savefig('./submission/imputation_comparison.png')  # Assure-toi que ce chemin est correct

plt.close()
