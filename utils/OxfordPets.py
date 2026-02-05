import os
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np

class OxfordPetsDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_head_crop=False, use_segmentation=False):
        """
        Args:
            root_dir (string): Percorso principale (es. 'res').
            transform (callable, optional): Trasformazioni.
            use_head_crop (bool): Se True, prova a ritagliare la testa usando i file XML.
            use_segmentation (bool): Se True, oscura lo sfondo usando le annotazioni fornite.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.use_head_crop = use_head_crop
        self.use_segmentation = use_segmentation
        
        self.images_dir = os.path.join(root_dir, 'images')
        self.xml_dir = os.path.join(root_dir, 'annotations', 'xmls')
        self.trimaps_dir = os.path.join(root_dir, 'annotations', 'trimaps')
        self.annotations_path = os.path.join(root_dir, 'annotations', 'list.txt')
        
        self.data = []
        self.targets = []
        
        if not os.path.exists(self.annotations_path):
             raise FileNotFoundError(f"Non trovo il file di annotazioni in: {self.annotations_path}")

        with open(self.annotations_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_name = parts[0]
                    class_id = int(parts[1])
                    label = class_id - 1
                    
                    self.data.append((image_name, label))
                    self.targets.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        
        img_path = os.path.join(self.images_dir, img_name + '.jpg')
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            # Se l'immagine non c'è, passiamo alla successiva
            return self.__getitem__((idx + 1) % len(self))
        
        # 2. Gestione Maschera (Segmentation)
        mask = None
        if self.use_segmentation:
            mask_path = os.path.join(self.trimaps_dir, img_name + '.png')
            if os.path.exists(mask_path):
                try:
                    # Le maschere sono immagini a un canale (L)
                    mask = Image.open(mask_path).convert('L')
                    # Ridimensioniamo la maschera se per caso non combacia con l'immagine originale
                    if mask.size != image.size:
                        mask = mask.resize(image.size, resample=Image.NEAREST)
                except:
                    mask = None # Se fallisce, niente maschera

        if self.use_head_crop:
            xml_path = os.path.join(self.xml_dir, img_name + '.xml')
            
            # Controlliamo se esiste l'annotazione per questa specifica immagine
            if os.path.exists(xml_path):
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    
                    """
                    Cerchiamo la bounding box. Nel file xlm sono presenti diversi tag annidati, ognuno che salva determinate infomazioni.
                    A noi interessano le coordinate spaziali dei due angoli del bounding box, quindi in totale 2 punti: (xmin, ymin), (xmax, ymax).
                    Analizzare solo quella parte aiuta il modello a non confondersi con sfondo ed eventuali altri elementi presenti nella foto, come 
                    persone, arredamento, sfondo.
                    """
                    obj = root.find('object')
                    if obj is not None:
                        bndbox = obj.find('bndbox')
                        if bndbox is not None:
                            xmin = int(bndbox.find('xmin').text)
                            ymin = int(bndbox.find('ymin').text)
                            xmax = int(bndbox.find('xmax').text)
                            ymax = int(bndbox.find('ymax').text)
                            
                            # Eseguiamo il ritaglio (Crop)
                            # Aggiungiamo un check per evitare crop vuoti o negativi
                            if xmax > xmin and ymax > ymin:
                                image = image.crop((xmin, ymin, xmax, ymax))

                                if mask is not None:
                                    mask = mask.crop((xmin, ymin, xmax, ymax))

                except Exception as e:
                    # Se l'XML è corrotto o il parsing fallisce, ignoriamo e 
                    # teniamo l'immagine intera. Non interrompiamo il training.
                    pass

        if self.use_segmentation and mask is not None:
            # Convertiamo in numpy per lavorare coi pixel
            np_img = np.array(image)
            np_mask = np.array(mask)
            
            """
            La segmentation si ottiene assegnando un valore diverso a seconda che il pixel appartenga all'animale, allo sfondo
            o ad un bordo che separa i due elementi.
            Per rimuovere lo sfondo, lo identifichiamo nella maschera, e assegnamo a zero il valore del pixel corrispondente
            nell'immagine originale.
            Così facendo si elimina rumore che altera la capacità di generalizzazione e la accuratezza delle previsioni del modello.
            """
            is_background = (np_mask != 1) 
            
            # Oscuriamo l'immagine: Dove è background, impostiamo il pixel a zero
            np_img[is_background] = 0
            
            # Riconvertiamo in PIL Image per passarla ai transforms
            image = Image.fromarray(np_img)

        if self.transform:
            image = self.transform(image)
            
        return image, label