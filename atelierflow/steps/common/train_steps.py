from atelierflow.core.model import Model
from typing import Any, Dict, Optional
from atelierflow.core.step_result import StepResult
import logging
from atelierflow.core.step import Step



class TrainNestedValidationModelStep(Step):
    def __init__(self, model: Model):
      self.model = model

    def run(self, input_data: Optional[StepResult], experiment_config: Dict[str, Any]) -> StepResult:
      if not input_data: 
        raise ValueError("TrainModelStep requires input data.")
      
      X_train, y_train = input_data.X, input_data.y
      logging.info(f"Fitting model '{self.model.__class__.__name__}'...")
