import torch.nn as nn

"""
Questo modello si basa sull'approccio MobileNet, che ottimizza le risorse separando il lavoro spaziale da quello sui canali.
Invece di usare un unico filtro pesante, l'operazione viene divisa in due step sequenziali:
Prima, una convoluzione spaziale 3x3 estrae le forme lavorando su ogni canale in modo indipendente (Depthwise). 
Successivamente, una convoluzione puntuale 1x1 si occupa di mescolare le informazioni attraverso la profondità della rete (Pointwise).
Questo approccio 'divide et impera' permette di ottenere performance simili a reti più grandi, ma con una frazione del costo computazionale.
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

class CustomMobileNet(nn.Module):
    def __init__(self, num_classes=37):
        super(CustomMobileNet, self).__init__()
        
        #Questo layer è quello che si interfaccia ai layers principali della rete
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        #Qui avvengono le convoluzioni descritte sopra, aumentando le dimensioni dei canali e riducendo quelle dell'immagine
        self.layers = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2), # Downsample
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=2), # Downsample
            DepthwiseSeparableConv(256, 256, stride=1),
            DepthwiseSeparableConv(256, 512, stride=2), # Downsample
            
            #Cuore della rete, le dimensioni ed i canali non variano
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            
            DepthwiseSeparableConv(512, 1024, stride=2), # Downsample finale
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
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)