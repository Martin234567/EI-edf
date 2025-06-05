import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os


# --- Chargement des données ----
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

#----- Préparation des données ------
TARGET = "Electricity_balance_not_controllable"
FEATURES = [
    "temperature", "nebulosity", "wind",
    "day_type_week", "day_type_jf", "day_type_week_jf",
    "period_holiday", "period_christmas", "period_summer",
    "nebulosity_by_solar_power_weights", "wind_by_wind_power_weights",
    "month", "tod", "toy", "week_number", "period_hour_changed"
]

X = train_df[FEATURES]
y = train_df[TARGET]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(test_df[FEATURES])

# --- Train / Val split ---
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# --- Conversion en tenseurs PyTorch ---
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# --- Dataset / DataLoader ---
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# --- Réseau de neurones ---
model_path = "mlp_model.pth"

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

model = MLP(len(FEATURES))
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# Entraînement MLP
if (os.path.exists(model_path)) and (input("Voulez-vous changez le nombre d'epoch ? ") in ["Non", "non","nn"]):
    print("📥 Chargement du modèle MLP...")
    model.load_state_dict(torch.load(model_path))
    model.eval()
else:
    nombre_epoch = int(input("Quel nombre d'epoch ? \n"))
    if os.path.exists(f"mlp_model_{nombre_epoch}.pth"):
        print("📥 Chargement du modèle MLP...")
        model.load_state_dict(torch.load(f"mlp_model_{nombre_epoch}.pth"))
        model.eval()
    else:
        print(f"Entrainement à {nombre_epoch} d'epoch")
        for epoch in range(nombre_epoch):
            model.train()
            for xb, yb in train_loader:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        torch.save(model.state_dict(), model_path)
        
# Évaluation sur validation
model.eval()
with torch.no_grad():
    val_preds = model(X_val_tensor)
    val_loss = loss_fn(val_preds, y_val_tensor)


# --- Prédiction test ---
model.eval()
with torch.no_grad():
    preds = model(X_test_tensor).squeeze().numpy()

# --- Sauvegarde du fichier de soumission ---

submission = pd.read_csv("test.csv")

#renomme la deuxième colonne de test.csv 
submission.columns.values[1] = "predicted"

# Remplace la colonne cible par 'predicted'
submission["predicted"] = preds

# Ne garde que 'Id' et 'predicted'
submission = submission[["Id", "predicted"]]

# Sauvegarde
submission.to_csv(f"submission{nombre_epoch}epoch.csv", index=False)

print("✅ Fichier de soumission 'submission.csv' avec colonnes 'Id' et 'predicted' prêt.")

