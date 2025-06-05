
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv('train.csv', sep=';')
df2 = pd.read_csv('test.csv', sep=';')

# Supprimer les colonnes inutiles
X_train = df.drop(columns=["date", "Date","Load","Electricity_balance_not_controllable","Solar_power","Wind_power"],errors='ignore')
X_test = df2.drop(columns=["date", "Date"],errors='ignore')
y = df["Load"]

# Diviser les données en ensembles d'entraînement et de test
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Créer et entraîner le modèle Random Forest
model1 = RandomForestRegressor(n_estimators=50, random_state=42)
model1.fit(X_train_scaled, y)

# Prédire sur l'ensemble de test
#y_pred = model.predict(X_test_scaled)

# Sauvegarder les prédictions dans un fichier CSV
#pd.DataFrame(y_pred, columns=["Predicted"]).to_csv("/Users/maximemacbook/Documents/Centrale/ST4_SCOOC/y_pred.csv", index=False, sep=',')

df_subset = X_train.iloc[189936:195744]
X_2024_scaled = scaler.fit_transform(df_subset)

pred_load_2024 = model1.predict(X_2024_scaled)

load_2024 = y.iloc[189936:195744]
# Évaluer le modèle
mse = mean_squared_error(load_2024, pred_load_2024)
r2 = r2_score(load_2024, pred_load_2024)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.3f}")

plt.plot(df["Id"][189936:195744],pred_load_2024)
plt.plot(df["Id"][189936:195744],load_2024)
plt.show()




#prédiction de solar power
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
model2 = RandomForestRegressor(n_estimators=50, random_state=42)
model2.fit(X_train_scaled, y)

# Prédire sur l'ensemble de test
#y_pred = model.predict(X_test_scaled)

# Sauvegarder les prédictions dans un fichier CSV
#pd.DataFrame(y_pred, columns=["Predicted"]).to_csv("/Users/maximemacbook/Documents/Centrale/ST4_SCOOC/y_pred.csv", index=False, sep=',')

# Créer un DataFrame avec la colonne Id indexée de 195744 à 213263
#ids = pd.Series(range(195744, 213264), name="Id")
#pred_df = pd.DataFrame({"Predicted_Solar_Power": y_pred})
##pred_df.insert(0, "Id", ids[:len(pred_df)])

# Sauvegarder les prédictions dans un fichier CSV
#pred_df.to_csv("/Users/maximemacbook/Documents/Centrale/ST4_SCOOC/y_pred_solar_power.csv", index=False, sep=',')

# Sauvegarder les prédictions dans un fichier CSV
#pred_df.to_csv("/Users/maximemacbook/Documents/Centrale/ST4_SCOOC/y_pred.csv", index=False, sep=',')

df_subset = X_train.iloc[189936:195744]
X_2024_scaled = scaler.fit_transform(df_subset)

pred_solar_2024 = model2.predict(X_2024_scaled)

solar_2024 = y.iloc[189936:195744]
# Évaluer le modèle
mse = mean_squared_error(solar_2024, pred_solar_2024)
r2 = r2_score(solar_2024, pred_solar_2024)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.3f}")

plt.plot(df["Id"][189936:195744],pred_solar_2024,'r')
plt.plot(df["Id"][189936:195744],solar_2024,'g')
plt.show()

# Charger les données
file_path1 = "train.csv"
file_path2 = "test.csv"
df = pd.read_csv(file_path1, sep=';')
df2 = pd.read_csv(file_path2, sep=';')

# Supprimer les colonnes inutiles
X_train = df.drop(columns=["date", "Date","Load","Electricity_balance_not_controllable","Solar_power","Wind_power"])
X_test = df2.drop(columns=["date", "Date"])
y = df["Wind_power"]

# Diviser les données en ensembles d'entraînement et de test
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Créer et entraîner le modèle Random Forest
model3 = RandomForestRegressor(n_estimators=50, random_state=42)
model3.fit(X_train_scaled, y)

# Prédire sur l'ensemble de test
#y_pred = model.predict(X_test_scaled)

# Sauvegarder les prédictions dans un fichier CSV
#pd.DataFrame(y_pred, columns=["Predicted"]).to_csv("/Users/maximemacbook/Documents/Centrale/ST4_SCOOC/y_pred.csv", index=False, sep=',')

# Créer un DataFrame avec la colonne Id indexée de 195744 à 213263
#ids = pd.Series(range(195744, 213264), name="Id")
#pred_df = pd.DataFrame({"Predicted_Wind_power": y_pred})
#pred_df.insert(0, "Id", ids[:len(pred_df)])

# Sauvegarder les prédictions dans un fichier CSV
#pred_df.to_csv("/Users/maximemacbook/Documents/Centrale/ST4_SCOOC/y_pred_wind_power.csv", index=False, sep=',')



df_subset = X_train.iloc[189936:195744]
X_2024_scaled = scaler.fit_transform(df_subset)

pred_wind_2024 = model3.predict(X_2024_scaled)

wind_2024 = y.iloc[189936:195744]
# Évaluer le modèle
mse = mean_squared_error(wind_2024, pred_wind_2024)
r2 = r2_score(wind_2024, pred_wind_2024)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.3f}")

plt.plot(df["Id"][189936:195744],pred_wind_2024,'r')
plt.plot(df["Id"][189936:195744],wind_2024,'g')
plt.show()


df = pd.read_csv(file_path1, sep=';')
df2 = pd.read_csv(file_path2, sep=';')

Elec = df["Electricity_balance_not_controllable"]
Elec_2024 = Elec.iloc[189936:195744]

# Charger les trois fichiers CSV
f1 = pd.read_csv("/Users/maximemacbook/Documents/Centrale/ST4_SCOOC/fichier_converti.csv") # Référence (ex: valeurs réelles)
f2 = pd.read_csv("/Users/maximemacbook/Documents/Centrale/ST4_SCOOC/y_pred_solar_power.csv") # Prédiction 1
f3 = pd.read_csv("/Users/maximemacbook/Documents/Centrale/ST4_SCOOC/y_pred_wind_power.csv") # Prédiction 2

# Extraire uniquement la 2ᵉ colonne de chaque fichier
col1 = f1.iloc[:, 1]
col2 = f2.iloc[:, 1]
col3 = f3.iloc[:, 1]

# Vérifier que les longueurs sont identiques
min_len = min(len(col1), len(col2), len(col3))
col1 = col1[:min_len]
col2 = col2[:min_len]
col3 = col3[:min_len]

# Calculer les différences
diff_1 = col1 - col2 - col3


# Créer un DataFrame avec les résultats
ids = pd.Series(range(195744, 195744 + min_len), name="Id")
result = pd.DataFrame({
"Id": ids,
"Predicted": diff_1
})

plt.plot(df["Id"][189936:195744],pred_load_2024-pred_solar_2024-pred_wind_2024,'r')
plt.plot(df["Id"][189936:195744],Elec_2024,'g')
plt.show()

mse = mean_squared_error(Elec_2024,pred_load_2024-pred_solar_2024-pred_wind_2024)
r2 = r2_score(Elec_2024,pred_load_2024-pred_solar_2024-pred_wind_2024)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.3f}")
print(pred_load_2024-pred_solar_2024-pred_wind_2024)
