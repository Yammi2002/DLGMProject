import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class OxfordPetsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Percorso della cartella principale del dataset (es. 'res').
                               Deve contenere le sottocartelle 'images' e 'annotations'.
            transform (callable, optional): Trasformazioni da applicare alle immagini.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'images')
        self.annotations_path = os.path.join(root_dir, 'annotations', 'list.txt')
        
        self.data = []
        
        if not os.path.exists(self.annotations_path):
             raise FileNotFoundError(f"Non trovo il file di annotazioni in: {self.annotations_path}")

        with open(self.annotations_path, 'r') as f:
            for line in f:
                # Ignoriamo le righe di commento che iniziano con #
                if line.startswith('#'):
                    continue
                
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_name = parts[0]
                    class_id = int(parts[1])

                    label = class_id - 1
                    
                    self.data.append((image_name, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        
        img_path = os.path.join(self.images_dir, img_name + '.jpg')
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:

            return self.__getitem__((idx + 1) % len(self))
        
        if self.transform:
            image = self.transform(image)
            
        return image, label