# Pet Breed Classification using custom CNNs and Fine-Tuning

Questo progetto esplora diverse tecniche di Deep Learning per il riconoscimento e la classificazione delle razze di cani e gatti, utilizzando il dataset **Oxford-IIIT Pet**.

L'obiettivo è produrre diversi modelli, sia custom che preaddesrati e modificati mediante fine-tuning, per riconoscere con la maggiore accuratezza possibile la razza degli animali presenti nelle immagini.

## Dataset

Il progetto utilizza il [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).
- **Categorie:** 37 razze diverse (Cani e Gatti).
- **Immagini:** Circa 7,900 immagini totali.

## Installazione

Clona il repository e installa le dipendenze necessarie.

```bash
git clone https://github.com/Yammi2002/DLGMProject.git
cd DLGMProject
pip install -r requirements.txt
```
## Preparazione del Dataset

Il progetto prevede l'utilizzo della cartella `res/` (resources) per contenere i file del dataset. 
Hai due opzioni per preparare i dati:

1. **Download Automatico:** Nel file `config.yaml`, imposta il flag `download_data: True`. Lo script si occuperà automaticamente di scaricare gli archivi ufficiali, estrarli e posizionare le cartelle `images` e `annotations` al posto giusto.
2. **Setup Manuale:** Se hai già scaricato il dataset, crea una cartella `res/` nella root del progetto e inserisci all'interno le due cartelle estratte, ottenendo questa configurazione:
   - `res/images/`: Contiene tutte le immagini `.jpg`.
   - `res/annotations/`: Contiene le sottocartelle `xmls` e `trimaps`.
  
## Configurazioni

Il comportamento dell'addestramento è controllato dal file **`config.yaml`**. Questo permette di modificare i parametri senza toccare il codice sorgente. Le configurazioni sono divise in tre categorie principali:

### 1. Iperparametri
Parametri numerici che definiscono la durata e la dinamica del training.

* **`batch_size`**: Numero di immagini processate contemporaneamente prima dell'aggiornamento dei pesi.
* **`learning_rate`**
* **`epochs`**: Numero totale di iterazioni complete sul dataset.
* **`patience`**: Numero di epoche consecutive senza miglioramenti nella *Validation Loss* prima di interrompere l'addestramento (Early Stopping).

### 2. Ottimizzazione
Parametri tecnici relativi all'algoritmo di discesa del gradiente.

* **`optimizer`**: L'algoritmo di ottimizzazione scelto.
  * `SGD`
  * `Adam`
  * `AdamW`
* **`weight_decay`**: Coefficiente di regolarizzazione L2 sui pesi.
* **`momentum`**: (Solo per SGD) Fattore di accelerazione, tipicamente impostato a `0.9`.
* **`nesterov`**: (Solo per SGD) Se `True`, abilita il Nesterov Momentum per una convergenza più stabile.
* **`scheduler`**: Politica di gestione del Learning Rate.
  * `None`
  * `OneCycleLR`
  * `ReduceLROnPlateau`
* **`scheduler_patience`**: Numero di epoche di tolleranza prima di ridurre il Learning Rate.
* **`label_smoothing`**: Coefficiente per ammorbidire le etichette reali, riducendo l'overfitting e migliorando la generalizzazione.

### 3. Flags (Opzioni Booleane)
Interruttori per attivare o disattivare funzionalità specifiche.

* **`use_head_crop`**: **[Importante]** Se `True`, utilizza le annotazioni XML (Bounding Box) per ritagliare e addestrare la rete solo sui volti degli animali, eliminando il rumore di fondo.
* **`use_segmentation`**: **[Importante]** Se `True`, utilizza le annotazioni trimaps per rimuovere lo sfondo dalle immagini.
* **`use_augmentation`**: Se `True`, applica trasformazioni casuali (flip, crop, color jitter) durante il training per migliorare la generalizzazione.
* **`fine_tuning`**:
  * `True`: Utilizza una rete pre-addestrata (Transfer Learning) e allena solo l'ultimo layer.
  * `False`: Allena l'architettura CustomCNN da zero.
* **`resume_training`**: Se `True`, il sistema cerca un checkpoint salvato con il nome specificato in `model_name` e riprende l'addestramento da quell'epoca.
* **`save_on_drive`**: Se `True`, salva i checkpoint su Google Drive (utilizzato per Colab); altrimenti salva in locale.
* **`download_data`**: Se `True`, scarica ed estrae il dataset Oxford-IIIT Pet all'avvio dell'esecuzione.
* **`model_name`**: Stringa identificativa per il salvataggio e caricamento dei file `.pth` (checkpoint).

## Struttura del progetto
DLGMProject/                                                                                                     
├── project.ipynb          # Notebook principale per il training, testing e visualizzazione                                           
├── config.yaml            # File di configurazione degli iperparametri                                                        
├── requirements.txt       # Dipendenze necessarie per l'esecuzione                                                            
├── README.md              # Documentazione del progetto                                                  
│      
├── res/                   # Cartella contenente i file originali del dataset                                 
│   ├── images/            # Immagini RGB (.jpg) di cani e gatti                                                       
│   └── annotations/       # File XML (bounding box), trimaps (segmentazione) e split liste                                                                
├── utils/                 # Moduli di utilità per gestione dati e configurazione                                                                   
│   ├── config.py          # Gestione modulare delle configurazioni e factory pattern                                                                            
│   ├── dataDownloader.py  # Script per il download e l'estrazione del dataset Oxford-IIIT Pet                                                                  
│   └── OxfordPets.py      # Gestione del Dataset, data augmentation e preprocessing                                                               
│                                                                                                                    
├── models/        # Architetture neurali sviluppate                                                                                      
│   ├── SimpleCNN.py       # Baseline ispirata a VGG (struttura semplice)                                                                                 
│   ├── Deep&Narrow.py     # CNN profonda con focus su pochi canali e Global Average Pooling                                                                       
│   ├── CustomResNet.py    # Implementazione ResNet-18 con Skip Connections                                                                                        
│   ├── InceptionModel.py  # Rete parallela ispirata a GoogLeNet                                                                                                
│   └── MobileCNN.py       # Rete efficiente basata su Depthwise Separable Convolutions 
│                                                                                                              
├── checkpoints/   # Dove vengono salvati i checkpoint degli addestramenti      
│                                                                                       
└── results/               # Output dell'addestramento (matrici di confusione e grafici)                                                              
    ├── SimpleCNN/                                                                                             
    ├── Deep&Narrow/                                                                                                       
    ├── CustomResNet/                                                                                                
    ├── InceptionModel/                                                                                                
    ├── Fine-tuning/                                                                                                   
    └── MobileCNN/                                                                                                        
