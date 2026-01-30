import torch
import torch.nn as nn

"""
Questo modello implementa un'architettura "Deep & Narrow" ispirata allo stile VGG.
La rete è composta da una sequenza di 10 strati convoluzionali (kernel 3x3) organizzati in 4 blocchi,
ognuno seguito da Max Pooling per la riduzione dimensionale.

Caratteristiche distintive:
1. Uso sistematico di Batch Normalization dopo ogni convoluzione per stabilizzare il gradiente in profondità.
2. Larghezza contenuta (max 256 canali) per limitare il costo computazionale.
3. Global Average Pooling (GAP) finale: riduce le feature spaziali a un singolo vettore per canale, 
   eliminando la necessità di strati densi (Fully Connected) intermedi pesanti e riducendo drasticamente i parametri.
Obiettivo: Massimizzare l'efficienza parametrica mantenendo una buona capacità di generalizzazione.
"""

class DeepNarrowCNN(nn.Module):
    def __init__(self, num_classes=37):
        super(DeepNarrowCNN, self).__init__()
        
        # Helper per creare un blocco Conv -> BN -> ReLU
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.block1 = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 32),
            nn.MaxPool2d(2, 2) 
        )

        self.block2 = nn.Sequential(
            conv_block(32, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2, 2) 
        )

        self.block3 = nn.Sequential(
            conv_block(64, 128),
            conv_block(128, 128),
            conv_block(128, 128),
            nn.MaxPool2d(2, 2)
        )

        self.block4 = nn.Sequential(
            conv_block(128, 256),
            conv_block(256, 256),
            conv_block(256, 256),
            nn.MaxPool2d(2, 2)
        )
        
        #Questo layer serve per ridimensionare il prodotto delle convoluzioni precedenti facendo la media dei valori per ciascun canale. Questo riduce i parametri
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(256, num_classes)
        
        self._initialize_weights()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)