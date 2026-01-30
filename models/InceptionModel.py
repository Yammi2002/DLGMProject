import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Questo modello si ispira all'architettura Inception (GoogLeNet).
La filosofia qui cambia radicalmente: invece di andare solo in profondità (Deep), andiamo in larghezza (Wide).

In ogni "Inception Module", l'input viene processato PARALLELAMENTE da filtri di dimensioni diverse:
1. Conv 1x1: Cattura relazioni tra i canali e riduce la dimensionalità.
2. Conv 3x3: Cattura dettagli medi.
3. Conv 5x5: Cattura feature più ampie e contestuali.
4. MaxPool: Mantiene le feature più forti.

Tutti questi risultati vengono CONCATENATI. La rete imparerà autonomamente se per classificare
quella specifica feature le serve guardare "da vicino" (3x3) o "da lontano" (5x5).
"""

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3, out_5x5):
        super(InceptionModule, self).__init__()
        
        # RAMO 1: Convoluzione 1x1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.BatchNorm2d(out_1x1),
            nn.ReLU(inplace=True)
        )
        
        # RAMO 2: Convoluzione 3x3 
        # Usiamo padding=1 per mantenere le dimensioni uguali all'input
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_3x3),
            nn.ReLU(inplace=True)
        )
        
        # RAMO 3: Convoluzione 5x5
        # Padding=2 per mantenere le dimensioni.
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_5x5),
            nn.ReLU(inplace=True)
        )
        
        # RAMO 4: Max Pooling
        # Seguito da 1x1 per aggiustare i canali se necessario (qui semplificato).
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_1x1, kernel_size=1), # Proiezione per ridurre canali
            nn.BatchNorm2d(out_1x1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Eseguiamo tutti i rami in parallelo
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y4 = self.branch4(x)
        
        # Concateniamo i risultati lungo la dimensione dei canali (dim=1)
        # L'output avrà canali = out_1x1 + out_3x3 + out_5x5 + out_1x1
        return torch.cat([y1, y2, y3, y4], 1)

class CustomInceptionNet(nn.Module):
    def __init__(self, num_classes=37):
        super(CustomInceptionNet, self).__init__()
        
        # Stem iniziale
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2) # Dimensione immagine 224 -> 112
        )

        # BLOCCO 1 
        self.inception1 = InceptionModule(in_channels=32, out_1x1=16, out_3x3=32, out_5x5=16) 
        # Output canali: 16+32+16+16 = 80 

        self.pool1 = nn.MaxPool2d(2, 2) # Dimensione immagine 112 -> 56

        # BLOCCO 2 
        self.inception2 = InceptionModule(in_channels=80, out_1x1=48, out_3x3=96, out_5x5=48)
        # Output canali: 48+96+48+48 = 240 

        self.pool2 = nn.MaxPool2d(2, 2) # Dimensione immagine 56 -> 28

        # BLOCCO 3
        self.inception3 = InceptionModule(in_channels=240, out_1x1=64, out_3x3=192, out_5x5=64)
        # Output canali: 64+192+64+64 = 384

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(384, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)
        
        x = self.inception1(x)
        x = self.pool1(x)
        
        x = self.inception2(x)
        x = self.pool2(x)
        
        x = self.inception3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x