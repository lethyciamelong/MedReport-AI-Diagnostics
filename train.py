import torch
import torch.optim as optim
import torch.nn as nn
from src.data_loader import MedReportDataset
from src.model import get_model
from torchvision import transforms
import pandas as pd

# 1. Hyperparamètres
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 1 # On commence par 1 pour tester

# 2. Charger les données (On utilise tes 2 Go d'images)
df = pd.read_csv('data/Data_Entry_2017.csv')


# --- AJOUTE CE BLOC ICI ---
import os
img_dir = 'data/images'
# On ne garde que les lignes où le fichier image existe physiquement sur le disque
df = df[df['Image Index'].apply(lambda x: os.path.exists(os.path.join(img_dir, x)))]
print(f"Nombre d'images trouvées localement : {len(df)}")
# --------------------------





# On s'assure d'avoir les labels (comme on a fait hier dans le test)
labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
for l in labels: df[l] = df['Finding Labels'].map(lambda x: 1 if l in x else 0)

# 3. Pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = MedReportDataset(df, 'data/images', transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 4. Modèle, Perte et Optimiseur
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=14).to(device)
criterion = nn.BCELoss() # Binary Cross Entropy pour le multi-label
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5. Boucle d'entraînement très simple
model.train()
for epoch in range(EPOCHS):
    for i, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Étape [{i}/{len(train_loader)}], Perte : {loss.item():.4f}")
    

# Créer le dossier s'il n'existe pas
os.makedirs('models', exist_ok=True)

# Sauvegarder les poids du modèle
torch.save(model.state_dict(), 'models/medreport_model.pth')
print("✅ Modèle sauvegardé avec succès dans models/medreport_model.pth")