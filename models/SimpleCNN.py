import torch.nn as nn

"""
MODELLO: Mini-VGG (CustomNN)
DESCRIZIONE:
    Questo modello rappresenta la baseline di partenza per i futuri esperimenti.
    Segue un'architettura stile VGG semplificata con 8 strati apprendibili (6 Conv + 2 FC).

CARATTERISTICHE PRINCIPALI:
    - Input: Immagini RGB (3 canali).
    - Backbone: 3 blocchi sequenziali, ognuno composto da doppia convoluzione 3x3, 
      Batch Normalization, ReLU e Max Pooling 2x2.
    - Bottleneck: Utilizza Global Average Pooling (AdaptiveAvgPool) invece del semplice Flatten.
      Questo riduce drasticamente il numero di parametri e previene l'overfitting spaziale.
    - Classificatore: MLP a due stadi (256 -> 512 -> num_classes) con Dropout (0.5).
"""

class CustomNN(nn.Module):
    def __init__(self, num_classes=37):
        super(CustomNN, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        #Questo layer serve per ridimensionare il prodotto delle convoluzioni precedenti facendo la media dei valori per ciascun canale. Questo riduce i parametri
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = self.avgpool(x)
        x = self.classifier(x)
        return x