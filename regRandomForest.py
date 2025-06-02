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
df = df.head(5000)
df2 = df2.head(5000)
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

# Évaluer le modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.6f}")

