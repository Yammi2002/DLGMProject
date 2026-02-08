import torch.nn as nn

"""
MODELLO: Custom MobileNet (ispirato dall'architettura MobileNet: Howard, A. G., et al. (2017) "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.")
DESCRIZIONE:
    Questo modello implementa un'architettura leggera ispirata a MobileNet V1.
    L'obiettivo è ottimizzare il rapporto performance/costo computazionale separando 
    l'elaborazione spaziale da quella sui canali.

PRINCIPIO DI FUNZIONAMENTO:
    Invece di usare filtri convoluzionali standard, ogni blocco opera in due fasi:
    1. Depthwise Conv (Spaziale): Filtra ogni canale di input singolarmente.
    2. Pointwise Conv (Canale): Combinazione lineare 1x1 dei canali.
    
CARATTERISTICHE ARCHITETTURALI:
    - Downsampling: Non utilizza MaxPool. La riduzione dimensionale avviene tramite 
      convoluzioni con stride=2.
    - Global Average Pooling: Riduce le dimensioni spaziali a 1x1 prima del classificatore,
      minimizzando i parametri del layer Dense finale.
"""

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        """
        Convloluzione Depthwise, il parametro groups di default è impostato ad 1,
        indicando in una convoluzione classica che i canali non sono da trattare separatamente,
        ma vanno considerati insieme per produrre l'output.
        Ponendolo uguale a in_channels non uniamo temporaneamente l'effetto dei vari canali.
        Dato questo, abbiamo che i canali di output di questo layer sono uguali a quelli di input,
        solo quando andiamo ad eseguire l'altra convoluzione andiamo a variare le dimensioni.
        """
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        """
        Convouzione Pointwise, sparisce il parametro groups (defaul è 1), indicando che ora
        le informazioni dei canali vanno unite.
        In questo momento si produce la variazione di dimensione impostando out_channels.
        """
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, 
                      padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CustomCNN(nn.Module):
    def __init__(self, num_classes=37):
        super(CustomCNN, self).__init__()
        
        #Questo layer è quello che si interfaccia ai layers principali della rete
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        #Qui avvengono le convoluzioni descritte sopra, aumentando le dimensioni dei canali e riducendo quelle dell'immagine
        self.layers = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
            DepthwiseSeparableConv(256, 512, stride=2),
            
            #Cuore della rete, le dimensioni ed i canali non variano
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            
            DepthwiseSeparableConv(512, 1024, stride=2),
            DepthwiseSeparableConv(1024, 1024, stride=1)
        )
        
        #Questo layer serve per ridimensionare il prodotto delle convoluzioni precedenti facendo la media dei valori per ciascun canale. Questo riduce i parametri
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        #Alcuni dei pesi vengono azzerati, per evitare overfitting (p: probabilità)
        self.dropout = nn.Dropout(p=0.2)

        #Layer finale per avere indicazione sulla previsione della rete
        self.fc = nn.Linear(1024, num_classes)
        
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.layers(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1) #Ridimensioniamo il tensore da passare al classificatore finale, passiamo da 4 dimensioni a 2
        x = self.dropout(x)
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)