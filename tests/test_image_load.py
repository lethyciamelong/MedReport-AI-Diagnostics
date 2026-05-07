import os
import sys
import torch
import pandas as pd
from torchvision import transforms

# 1. On ajoute le dossier 'src' pour pouvoir importer ton loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import MedReportDataset

def test_loading():
    # Chemins (à adapter selon ton ThinkPad)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(BASE_DIR, 'data', 'Data_Entry_2017.csv')
    img_dir = os.path.join(BASE_DIR, 'data', 'images')

    # Chargement du CSV pour le test
    df = pd.read_csv(csv_path)
    
    # AJOUTE CECI : Création des colonnes de labels pour le test
    labels = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
        'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
        'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]
    for label in labels:
        df[label] = df['Finding Labels'].map(lambda x: 1 if label in x else 0)


    # On définit les transformations (ce que tu as déjà écrit)
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialisation du dataset
    try:
        dataset = MedReportDataset(dataframe=df, img_dir=img_dir, transform=data_transforms)
        
        # On essaie de charger la TOUTE PREMIÈRE image du CSV
        image, label = dataset[0]
        
        print("-----------------------------------------")
        print("✅ TEST RÉUSSI : Le pipeline fonctionne !")
        print(f"Dimension du tenseur image : {image.shape}") # Devrait afficher [3, 224, 224]
        print(f"Type de l'image : {type(image)}")
        print("-----------------------------------------")
        
    except Exception as e:
        print("-----------------------------------------")
        print(f"❌ TEST ÉCHOUÉ : {e}")
        print("Vérifie que l'image du CSV existe bien dans data/images/")
        print("-----------------------------------------")

if __name__ == "__main__":
    test_loading()