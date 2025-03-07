import pandas as pd

# Charger les données d'entrée (données de test imputées)
test_data = pd.read_csv('data/test.csv')

# Charger les résultats d'imputation (ici, test_data_imputed)
test_data_imputed = pd.read_csv('submission/submission.csv')  # ou à partir du processus d'imputation

# Identifier les indices des valeurs manquantes
missing_indices = test_data[test_data.isnull()].index

# Créer un DataFrame avec les indices et les valeurs imputées
submission_df = pd.DataFrame({
    'ID': missing_indices,
    'TARGET': test_data_imputed.iloc[missing_indices]
})

# Sauvegarder dans un fichier CSV
submission_df.to_csv('submission/submission.csv', index=False)

print("Fichier de soumission créé : 'submission/submission.csv'")
