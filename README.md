# Pet Breed Classification: Custom CNNs vs Fine-Tuning

Questo progetto esplora diverse tecniche di Deep Learning per il riconoscimento e la classificazione delle razze di cani e gatti, utilizzando il dataset **Oxford-IIIT Pet**.

L'obiettivo è produrre diversi modelli, sia custom che preaddesrati e modificati mediante fine-tuning, per riconoscere con la maggiore accuratezza possibile quale razza sia presente nelle immagini.

## Dataset

Il progetto utilizza il [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).
- **Categorie:** 37 razze diverse (Cani e Gatti).
- **Immagini:** Circa 7,900 immagini totali.

## Installazione

Clona il repository e installa le dipendenze necessarie.

```bash
git clone https://github.com/Yammi2002/DLGMProject.git
cd pet-classification-cnn
pip install -r requirements.txt
```

## Configurazioni

È presente un file di configurazione: **config.yaml** che permette di configurare l'esecuzione del programma, impostando una serie di parametri.

* **Iperparametri**
  * `batch_size`
  * `learning_rate`
  * `epochs`
  * `patience`: indica il numero di epoche consecutive in cui non si vede un miglioramento della validation loss, serve per l'Early stopping
* **Training**
  * **Loss Function:** Categorical Crossentropy
  * **Optimizer:** Adam, SGD, AdamW
* **Flag**
  * `fine_tuning`: se impostato a True si utilizza il modello preaddestrato RasNet in cui si alterza solo l'ultimo layer
  * `use_augmentation`: se impostato a True applica delle trasformazioni al dataset di training per aumentare la generalizzazione che il modello farà dei dati
  * `resume_training`: se impostato a True viene recuperato il modello identificato da `model_name` e si riprende il suo addetsramento
  * `download_data`: se impostato a True il dataset verrà scaricato nella cartella "/res", nel caso sia già stato precedentemnete installato è possibile lasciare questo parametro come False
  * `model_name`
