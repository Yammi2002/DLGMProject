from dataclasses import dataclass
from enum import Enum
import yaml
import torch
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

@dataclass
class Config:
    
    model_name: str = "Improved_CNN"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    epochs: int = 30
    patience: int = 7
    
    learning_rate: float = 0.001
    optimizer: OptimizerType = OptimizerType.ADAM
    
    # Parametri specifici Optimizer
    weight_decay: float = 1e-4  
    momentum: float = 0.9       
    nesterov: bool = True   

    # --- Loss ---
    loss: LossType = LossType.CROSS_ENTROPY
    label_smoothing: float = 0.1 

    # --- Flags ---
    fine_tuning: bool = False
    use_augmentation: bool = True
    resume_training: bool = False
    download_data: bool = False
    
    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
            
        valid_keys = {k: v for k, v in yaml_data.items() if k in cls.__annotations__}

        try:
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
            return optim.Adam(model_parameters, lr=self.learning_rate, 
                              weight_decay=self.weight_decay)
        
        elif self.optimizer == OptimizerType.ADAMW:
            return optim.AdamW(model_parameters, lr=self.learning_rate, 
                               weight_decay=self.weight_decay)
        
        elif self.optimizer == OptimizerType.SGD:
            return optim.SGD(model_parameters, lr=self.learning_rate, 
                             momentum=self.momentum, 
                             weight_decay=self.weight_decay,
                             nesterov=self.nesterov)
        
        raise ValueError(f"Optimizer {self.optimizer} non implementato")

    def init_scheduler(self, optimizer):
        """Inizializza lo scheduler (opzionale)."""
        if self.scheduler == SchedulerType.NONE:
            return None
        
        if self.scheduler == SchedulerType.PLATEAU:
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', 
                factor=self.scheduler_factor, 
                patience=self.scheduler_patience, 
                min_lr=self.min_lr,
                verbose=True
            )
        
        if self.scheduler == SchedulerType.ONE_CYCLE:
            pass 

        return None