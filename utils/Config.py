from dataclasses import dataclass
from enum import Enum
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

# Classi emum per gestire le conf possibili
class LossType(Enum):
    CROSS_ENTROPY = "CrossEntropy"

class OptimizerType(Enum):
    ADAM = "Adam"
    SGD = "SGD"
    ADAMW = "AdamW"

@dataclass
class Config:
    # Percorsi
    data_path: str = "./res"
    checkpoint_dir: str = "./checkpoints"
    
    # Iperparametri
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 10
    patience: int = 3
    
    loss: LossType = LossType.CROSS_ENTROPY
    optimizer: OptimizerType = OptimizerType.ADAM
    
    # Flag
    fine_tuning: bool = False
    use_augmentation: bool = True
    resume_training: bool = False
    download_data: bool = False

    model_name: str = "Default_Exp"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
            
        # Filtriamo le chiavi extra che non esistono nella classe
        valid_keys = {k: v for k, v in yaml_data.items() if k in cls.__annotations__}

        try:
            if 'loss' in valid_keys:
                valid_keys['loss'] = LossType(valid_keys['loss'])
            
            if 'optimizer' in valid_keys:
                valid_keys['optimizer'] = OptimizerType(valid_keys['optimizer'])
                
        except ValueError as e:
            raise ValueError(f"ERRORE: Valore non valido nel YAML. {e}")

        return cls(**valid_keys)