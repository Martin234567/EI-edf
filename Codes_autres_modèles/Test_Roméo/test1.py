import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd # Assurez-vous que pandas est importé pour le reste du code

# --- Réseau de neurones ---
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        # Utilisation des paramètres fournis
        hidden_size = 128 # Taille de la première couche cachée
        dropout_rate = 0.16955393328907425 # Taux de dropout

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size), # input_dim -> 128
            nn.BatchNorm1d(hidden_size),
            nn.ELU(),
            nn.Dropout(dropout_rate), # Taux de dropout appliqué ici
            nn.Linear(hidden_size, 64), # 128 -> 64 (conservé car c'est une réduction courante)
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(dropout_rate), # Taux de dropout appliqué ici aussi
            nn.Linear(64, 1) # 64 -> sortie unique
        )

    def forward(self, x):
        return self.model(x)

# --- Paramètres d'entraînement ---
# Ces variables doivent être définies AVANT la création du DataLoader et de l'Optimiseur
BATCH_SIZE = 32
LEARNING_RATE = 0.00027715197705355124

# --- Exemple d'utilisation (doit être intégré dans votre script principal) ---

# Assurez-vous que vos DataFrames (df_train, df_val, df_test) sont prêts
# et que les features ont été créées (y compris les moyennes mobiles et les NaNs gérés)
# Exemple de données fictives si vous testez ce bloc séparément
# Supposons que X_train, y_train, X_val, y_val, X_test sont déjà définis comme des DataFrames/arrays
# comme après votre preprocessing et feature engineering.

# Conversion en Tensors PyTorch
# Ces lignes sont cruciales et doivent venir après tout le feature engineering
# Assurez-vous que les DataFrames X_train, y_train, etc. sont correctement formatés
# et que les NaNs sont gérés AVANT cette étape.
# input_dim doit être calculé après avoir ajouté toutes les features.
try:
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).reshape(-1, 1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
except NameError:
    print("Veuillez vous assurer que X_train, y_train, X_val, y_val, X_test sont définis et préparés.")
    # Exemples de données pour que le code puisse s'exécuter si les DF ne sont pas encore définis
    input_dim_dummy = 10
    X_train_tensor = torch.randn(100, input_dim_dummy)
    y_train_tensor = torch.randn(100, 1)
    X_val_tensor = torch.randn(20, input_dim_dummy)
    y_val_tensor = torch.randn(20, 1)
    X_test_tensor = torch.randn(30, input_dim_dummy)


# Définition de input_dim basée sur vos données
input_dim = X_train_tensor.shape[1]

# Instanciation du modèle
model = MLP(input_dim)

# DataLoaders avec le nouveau BATCH_SIZE
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Optimiseur avec le nouveau LEARNING_RATE
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss() # Supposons que vous utilisez toujours MSE

# Scheduler (conserver ReduceLROnPlateau est une bonne pratique)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

# Boucle d'entraînement (inchangée par rapport à la dernière version)
num_epochs = 50
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_val_loss += loss.item()

    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    scheduler.step(avg_val_loss)

    print(f'Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}')

# Tracer les Losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss with Tuned Parameters')
plt.legend()
plt.grid(True)
plt.show()

# Prédictions sur l'ensemble de test
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_predictions = test_predictions.numpy()

# ... (Le reste de votre code pour générer la soumission, par exemple)
# submission = pd.read_csv("sample_submission.csv")
# submission['Electricity_balance_not_controllable'] = test_predictions
# submission.to_csv("my_submission.csv", index=False)