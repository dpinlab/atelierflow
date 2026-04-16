from atelierflow.core.model import Model
from typing import Any, Dict, Optional
import logging
from atelierflow.core.step import Step
import numpy as np
from dataclasses import dataclass

@dataclass
class TrainingData:
    X: np.ndarray
    y: np.ndarray

@dataclass
class TrainedModelResult:
    model: Model
    train_time: float

# 2. O Step usando os tipos definidos acima
class TrainNestedValidationModelStep(Step[TrainingData, TrainedModelResult]):
    def __init__(self, model: Model):
        self.model = model

    def run(self, input_data: TrainingData, experiment_config: Dict[str, Any]) -> TrainedModelResult:
        X_train, y_train = input_data.X, input_data.y
        
        logging.info(f"Fitting model '{self.model.__class__.__name__}'...")
        
        return TrainedModelResult(model=self.model, train_time=10.5)