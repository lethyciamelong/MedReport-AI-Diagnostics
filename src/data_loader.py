import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class MedReportDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.labels = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
            'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
            'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx]['Image Index'])
        image = Image.open(img_name).convert('RGB')
        
        # On extrait les valeurs des colonnes de maladies
        labels = torch.tensor(self.df.iloc[idx][self.labels].values.astype(float), dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, labels