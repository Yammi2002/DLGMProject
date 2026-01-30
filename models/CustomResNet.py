import torch.nn as nn
import torch.nn.functional as F

"""
Questo modello si basa sull'approccio RasNet, che riesce ad avere un elevato numero di layer senza che si verifichi il fenomeno
del vanishing del gradiente. Aummentando la profondità si è osservato come in strutture come la classica VCC si andasse incontro ad
un aumento dell'errore di training e di validation. Questo fenomeno differisce dall'overfitting, in quanto si osserverebbe un aumento
solo nel momento di validazione mentre osserverebbe una diminuzione dell'errore di training.

La soluzione a questo fenomeno risiede nelle skip connections, in cui a termine di un blocco convolutivo si aggiungono informazioni 
sull'input allo stesso, evitando che si perda informazione all'aumentare della prodondità.
"""

class ResidualBlock(nn.Module):
    """
    Il mattoncino fondamentale della ResNet.
    Contiene 2 convoluzioni e la skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut (Skip Connection) inizialmente vuota
        self.shortcut = nn.Sequential()

        """
        Questo controllo è essenziale in quanto la dimensione dell'input e dell'output di un blocco convolutivo in cui si utilizza
        una skip connection deve essere della stessa dimensione.
        Quando si applica uno stride diverso da 1 per andare a decremnetare la dimensione dell'immagine ed aumentare il numero dei
        canali non è possibile applicare una skip connection.        
        """
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x 
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Qui avviene la skip connection, sommo l'informazione dell'input
        out += self.shortcut(identity) 
        
        out = F.relu(out)
        return out

class CustomResNet18(nn.Module):
    def __init__(self, num_classes=37):
        super(CustomResNet18, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(in_c=64, out_c=64, stride=1)
        
        self.layer2 = self._make_layer(in_c=64, out_c=128, stride=2)
        
        self.layer3 = self._make_layer(in_c=128, out_c=256, stride=2)
        
        self.layer4 = self._make_layer(in_c=256, out_c=512, stride=2)
        
        #Questo layer serve per ridimensionare il prodotto delle convoluzioni precedenti facendo la media dei valori per ciascun canale. Questo riduce i parametri
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)) 

        self.fc = nn.Linear(512, num_classes)     

    def _make_layer(self, in_c, out_c, stride):
        """
        Funzione helper per creare un blocco composto da 2 layer
        """
        layers = []
        layers.append(ResidualBlock(in_c, out_c, stride))
        layers.append(ResidualBlock(out_c, out_c, stride=1))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.layer4(x) 
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        
        return x