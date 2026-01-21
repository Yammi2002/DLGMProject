import os
import shutil
import tarfile
import urllib.request

def download_data():
    
    """
    Scarica, estrae e organizza il dataset Oxford-IIIT Pet.
    """
    
    # Rimuoviamo le caretelle se già presenti per avere il dataset completo
    folders_to_clean = ['images', 'annotations', 'res']
    for folder in folders_to_clean:
        if os.path.exists(folder):
            print(f"Pulizia cartella vecchia: {folder}...")
            shutil.rmtree(folder)
            
    # Crea la cartella di destinazione
    os.makedirs('res', exist_ok=True)

    # URL dei file
    urls = {
        'images': 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz',
        'annotations': 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz'
    }

    # Scaricamento ed Estrazione
    for name, url in urls.items():
        filename = f"{name}.tar.gz"
        
        if not os.path.exists(filename):
            urllib.request.urlretrieve(url, filename)
        
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall()
            
        # Spostamento in 'res'
        source_dir = name
        dest_dir = os.path.join('res', name)
        
        if os.path.exists(source_dir):
            shutil.move(source_dir, dest_dir)
    
    # Pulizia
    if os.path.exists('images.tar.gz'): os.remove('images.tar.gz')
    if os.path.exists('annotations.tar.gz'): os.remove('annotations.tar.gz')

if __name__ == "__main__":
    download_data()