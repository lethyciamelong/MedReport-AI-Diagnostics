import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=14):
    # Charger un modèle pré-entraîné sur ImageNet
    model = models.densenet121(weights='IMAGENET1K_V1')
    
    # Modifier la dernière couche
    num_ftrs = model.classifier.in_features
    
    # On remplace le classifieur par une couche adaptée à nos 14 pathologies
    # On utilise Sigmoid car un patient peut avoir PLUSIEURS maladies en même temps
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, num_classes),
        nn.Sigmoid() 
    )
    return model