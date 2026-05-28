from dataclasses import dataclass
from enum import Enum
import yaml
import torch.nn as nn
import torch.optim as optim

# --- ENUMS ---
class LossType(Enum):
    CROSS_ENTROPY = "CrossEntropy"

class OptimizerType(Enum):
    ADAM = "Adam"
    SGD = "SGD"
    ADAMW = "AdamW"

class SchedulerType(Enum):
    NONE = "None"
    PLATEAU = "ReduceLROnPlateau"
    ONE_CYCLE = "OneCycleLR"

"""
Questa classe consente al programma di essere modulare, recuperando le configurazioni 
direttamente da un file YAML.
Utilizza metodi factory per istanziare ottimizzatori e scheduler da passare al training loop.
"""

@dataclass
class Config:
    
    # Identità esperimento
    model_name: str = "Improved_CNN"
    resume_training: bool = False
    save_on_drive: bool = False

    # Dati e trasformazioni
    batch_size: int = 32
    download_data: bool = False
    use_augmentation: bool = True
    use_head_crop: bool = False
    use_segmentation: bool = False

    # Modello e task
    fine_tuning: bool = False
    loss: LossType = LossType.CROSS_ENTROPY

    # Training
    epochs: int = 30
    patience: int = 7

    # Ottimizzatore
    learning_rate: float = 0.001
    optimizer: OptimizerType = OptimizerType.ADAM
    weight_decay: float = 1e-4  
    momentum: float = 0.9       
    nesterov: bool = True   

    # Loss
    label_smoothing: float = 0.1 

    # Scheduler
    scheduler: SchedulerType = SchedulerType.NONE
    scheduler_patience: int = 3
    
    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
            
        # Filtra solo le chiavi che esistono nella dataclass
        valid_keys = {k: v for k, v in yaml_data.items() if k in cls.__annotations__}

        try:
            # Conversione stringa -> Enum
            if 'loss' in valid_keys:
                valid_keys['loss'] = LossType(valid_keys['loss'])
            if 'optimizer' in valid_keys:
                valid_keys['optimizer'] = OptimizerType(valid_keys['optimizer'])
            if 'scheduler' in valid_keys:
                valid_keys['scheduler'] = SchedulerType(valid_keys['scheduler'])
                
        except ValueError as e:
            raise ValueError(f"ERRORE YAML: Valore enum non valido. {e}")

        return cls(**valid_keys)


    def get_loss_fn(self):
        """Restituisce l'istanza della Loss configurata."""
        if self.loss == LossType.CROSS_ENTROPY:
            return nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        raise ValueError(f"Loss {self.loss} non implementata")

    def init_optimizer(self, model_parameters):
        """Inizializza l'ottimizzatore passando i parametri del modello."""
        
        if self.optimizer == OptimizerType.ADAM:
            """
            Combina informazioni sulla media dei gradienti passati con quella della media delle varianze passate.
            A differenza di SGD, adatta il Learning Rate per ogni singolo parametro: rallenta 
            i parametri che oscillano molto e accelera quelli stabili.
            """
            return optim.Adam(model_parameters, lr=self.learning_rate, 
                              weight_decay=self.weight_decay)
        
        elif self.optimizer == OptimizerType.ADAMW:
            """
            Variante moderna di Adam che applica la weight decay (regolarizzazione) 
            direttamente sui pesi e non sul gradiente.
            Questo disaccoppiamento migliora drasticamente la generalizzazione rispetto ad Adam classico.
            """
            return optim.AdamW(model_parameters, lr=self.learning_rate, 
                               weight_decay=self.weight_decay)
        
        elif self.optimizer == OptimizerType.SGD:
            """
            Utilizza il calcolo dei gradienti per aggiornare i pesi. Il momentum aggiunge 'inerzia' accumulando velocità 
            nella direzione corretta, permettendo di superare minimi locali e ridurre oscillazioni. 
            La correzione di Nesterov migliora il momentum calcolando il gradiente non nella posizione attuale, 
            ma nella posizione futura prevista, aumentando la stabilità.
            """
            # NOTA: Nesterov richiede momentum > 0
            momentum_val = self.momentum if self.momentum > 0 else 0.9 
            
            return optim.SGD(model_parameters, lr=self.learning_rate, 
                             momentum=momentum_val, 
                             weight_decay=self.weight_decay,
                             nesterov=self.nesterov)
        
        raise ValueError(f"Optimizer {self.optimizer} non implementato")

    def init_scheduler(self, optimizer, steps_per_epoch=None):
        """
        Inizializza lo scheduler.
        :param optimizer: L'ottimizzatore della rete.
        :param steps_per_epoch: Necessario per OneCycleLR.
        """
        if self.scheduler == SchedulerType.NONE:
            return None
        
        if self.scheduler == SchedulerType.PLATEAU:
            """
            ReduceLROnPlateau.
            Osserva la validation loss: se non migliora per 'patience' epoche,
            riduce il learning rate per scendere nel minimo locale con maggiore precisione.
            """
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',  
                patience=self.scheduler_patience
            )
        
        if self.scheduler == SchedulerType.ONE_CYCLE:
            """
            Altera il LR disegnando un'onda: parte basso, sale veloce al massimo, 
            e poi scende quasi a zero.
            - La salita veloce aiuta a superare i minimi locali instabili (regolarizzazione).
            - La discesa finale permette la super-convergenza.
            Richiede la chiamata al metodo .step() ad ogni bathc, non ad ogni epoca.
            """
            if steps_per_epoch is None:
                raise ValueError("Per utilizzare OneCycleLR devi passare 'steps_per_epoch' (len(train_loader)) a init_scheduler!")
            
            return optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate, # Usa il LR del config come picco massimo
                epochs=self.epochs,
                steps_per_epoch=steps_per_epoch
            )

        return None