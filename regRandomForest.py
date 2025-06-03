import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Charger les données
file_path1 = "/Users/maximemacbook/Documents/Centrale/ST4_SCOOC/train CSV.csv"
file_path2 = "/Users/maximemacbook/Documents/Centrale/ST4_SCOOC/données test CSV.csv"
df = pd.read_csv(file_path1, sep=';')
df2 = pd.read_csv(file_path2, sep=';')

# Supprimer les colonnes inutiles
X_train = df.drop(columns=["date", "Date","Load","Electricity_balance_not_controllable","Solar_power","Wind_power"])
X_test = df2.drop(columns=["date", "Date"])
y = df["Load"]

# Diviser les données en ensembles d'entraînement et de test
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Créer et entraîner le modèle Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y)

# Prédire sur l'ensemble de test
y_pred = model.predict(X_test_scaled)

# Sauvegarder les prédictions dans un fichier CSV
pd.DataFrame(y_pred, columns=["Predicted"]).to_csv("/Users/maximemacbook/Documents/Centrale/ST4_SCOOC/y_pred.csv", index=False, sep=',')

# Évaluer le modèle
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.3f}")

#%% prédiction de solar power


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Charger les données
file_path1 = "/Users/maximemacbook/Documents/Centrale/ST4_SCOOC/train CSV.csv"
file_path2 = "/Users/maximemacbook/Documents/Centrale/ST4_SCOOC/données test CSV.csv"
df = pd.read_csv(file_path1, sep=';')
df2 = pd.read_csv(file_path2, sep=';')

# Supprimer les colonnes inutiles
X_train = df.drop(columns=["date", "Date","Load","Electricity_balance_not_controllable","Solar_power","Wind_power"])
X_test = df2.drop(columns=["date", "Date"])
y = df["Solar_power"]

# Diviser les données en ensembles d'entraînement et de test
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Créer et entraîner le modèle Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y)

# Prédire sur l'ensemble de test
y_pred = model.predict(X_test_scaled)

# Sauvegarder les prédictions dans un fichier CSV
pd.DataFrame(y_pred, columns=["Predicted"]).to_csv("/Users/maximemacbook/Documents/Centrale/ST4_SCOOC/y_pred.csv", index=False, sep=',')

# Créer un DataFrame avec la colonne Id indexée de 195744 à 213263
ids = pd.Series(range(195744, 213264), name="Id")
pred_df = pd.DataFrame({"Predicted_Solar_Power": y_pred})
pred_df.insert(0, "Id", ids[:len(pred_df)])

# Sauvegarder les prédictions dans un fichier CSV
pred_df.to_csv("/Users/maximemacbook/Documents/Centrale/ST4_SCOOC/y_pred_solar_power.csv", index=False, sep=',')

# Sauvegarder les prédictions dans un fichier CSV
#pred_df.to_csv("/Users/maximemacbook/Documents/Centrale/ST4_SCOOC/y_pred.csv", index=False, sep=',')

# Évaluer le modèle
#mse = mean_squared_error(y, y_pred)
#r2 = r2_score(y, y_pred)
#print(f"Mean Squared Error: {mse:.2f}")
#print(f"R2 Score: {r2:.3f}")

# %%
print("hello world")

# %%
