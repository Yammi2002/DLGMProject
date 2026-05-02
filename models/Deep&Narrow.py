import torch
import torch.nn as nn

"""
MODELLO: Deep & Narrow CNN (Custom VGG-Style)
DESCRIZIONE:
    A differenza delle architetture classiche molto larghe, questa rete punta sulla profondità 
    (10 layer convoluzionali) mantenendo il numero di canali contenuto (max 256).

CARATTERISTICHE ARCHITETTURALI:
    - Stack Convoluzionale: 4 blocchi progressivi. I primi due contengono 2 convoluzioni, 
      gli ultimi due ne contengono 3. Totale: 10 strati di feature extraction.
    - Efficienza: L'uso del Global Average Pooling finale elimina la necessità di 
      pesanti layer Fully Connected intermedi, riducendo drasticamente i parametri 
      e prevenendo l'overfitting.
    - Stabilizzazione: Batch Normalization sistematica dopo ogni conv per permettere 
      al gradiente di fluire attraverso i 10 strati senza svanire.
"""

class CustomCnn(nn.Module):
    def __init__(self, num_classes=37):
        super(CustomCnn, self).__init__()
        
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
        """
        Inizializza i pesi del modello seguendo le best practice per reti deep.
        Viene fatto sia per i layer convoluzionali che per i layer di batch normalization.
        Lo scopo è quello di ridurre l'effetto di scomparsa del gradiente, che solitamente si verifica quando aumenta
        la profondità della rete.
        Segnale di questo fenomeno è l'innalzamento dell'errore di training e di quello di validazione.
        """ 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)