import torch
import torch.nn as nn

"""
MODELLO: Inception CNN (Ispirato all'architettura GoogLeNet: Szegedy et al. (2015) "Going Deeper with Convolutions")
DESCRIZIONE:
    A differenza delle reti lineari classiche, questo modello punta sulla 
    parallelizzazione del calcolo per estrarre informazioni a diverse scale spaziali 
    contemporaneamente nello stesso livello della rete.

CARATTERISTICHE ARCHITETTURALI:
    - Inception Modules (7 blocchi): Il cuore della rete è composto da 7 moduli 
      Inception. Ogni modulo processa l'input attraverso 4 rami paralleli 
      (Conv 1x1, 3x3, 5x5 e MaxPool) le cui uscite vengono concatenate.
    - Profondità Totale: La rete conta 31 layer con parametri apprendibili 
      (30 convoluzioni e 1 fully connected), strutturati in uno "Stem" iniziale, 
      3 blocchi di Inception e un classificatore finale.
    - Feature Extraction: L'uso di kernel di dimensioni differenti 
      permette alla rete di catturare sia dettagli fini che pattern contestuali 
      più ampi, adattandosi a oggetti di dimensioni variabili nel dataset.
    - Efficienza e Regolarizzazione: L'uso dell'Adaptive Average Pooling riduce 
      drasticamente il numero di parametri finali, mentre il Dropout al 40% 
      previene l'overfitting nonostante l'elevata larghezza della rete.
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
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_3x3),
            nn.ReLU(inplace=True)
        )
        
        # RAMO 3: Convoluzione 5x5
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_5x5),
            nn.ReLU(inplace=True)
        )
        
        # RAMO 4: Max Pooling
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.BatchNorm2d(out_1x1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y4 = self.branch4(x)

        """
        Questa è la particolarità dell'architettura. Quando viene chiamato il metodo forward, vengono applicati kernel di dimensioni
        differenti sullo stesso input e il loro output viene concatenato in un unico tensore.
        """
        return torch.cat([y1, y2, y3, y4], 1)

class CustomCNN(nn.Module):
    def __init__(self, num_classes=37):
        super(CustomCNN, self).__init__()
        
        # --- STEM ---
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            nn.Conv2d(64, 192, kernel_size=3, padding=1), 
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # --- BLOCCO 1 ---
        # Input: 192. 
        # Output calc: 64 (br1) + 128 (br2) + 32 (br3) + 64 (br4 usa out_1x1) = 288
        self.inception1a = InceptionModule(192, out_1x1=64, out_3x3=128, out_5x5=32)
        
        # Input: 288 (era 256 nel tuo codice, corretto a 288).
        # Output calc: 128 + 192 + 96 + 128 = 544
        self.inception1b = InceptionModule(288, out_1x1=128, out_3x3=192, out_5x5=96)
        
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        # --- BLOCCO 2 ---
       # Input: 544 -> Output calc: 192 + 208 + 48 + 192 = 640
        self.inception2a = InceptionModule(544, out_1x1=192, out_3x3=208, out_5x5=48)
        # Input: 640 -> Output calc: 192 + 256 + 64 + 192 = 704
        self.inception2b = InceptionModule(640, out_1x1=192, out_3x3=256, out_5x5=64)
        # Input: 704 -> Output calc: 224 + 288 + 64 + 224 = 800
        self.inception2c = InceptionModule(704, out_1x1=224, out_3x3=288, out_5x5=64)
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)

        # --- BLOCCO 3 --- 
        # Input: 800 -> Output calc: 256 + 320 + 128 + 256 = 960
        self.inception3a = InceptionModule(800, out_1x1=256, out_3x3=320, out_5x5=128)
        # Input: 960 -> Output calc: 384 + 384 + 128 + 384 = 1280
        self.inception3b = InceptionModule(960, out_1x1=384, out_3x3=384, out_5x5=128)

        # --- CLASSIFICATORE ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4) # Vista la profondità delle rete aggiungiamo un dropout per evitare overfitting
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)
        
        x = self.inception1a(x)
        x = self.inception1b(x)
        x = self.pool1(x) 
        
        x = self.inception2a(x)
        x = self.inception2b(x)
        x = self.inception2c(x)
        x = self.pool2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x