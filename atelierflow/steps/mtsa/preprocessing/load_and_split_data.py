import logging
from mtsa.utils.utils import files_train_test_split
from typing import Any, Dict, Optional
from atelierflow.core.step import Step
from dataclasses import dataclass
import numpy as np


@dataclass
class SplitDataResult:
  X_train: np.ndarray
  X_test: np.ndarray
  y_train: np.ndarray
  y_test: np.ndarray

class LoadAndSplitDataStep(Step[None, SplitDataResult]):
  """
  A preprocessing step that loads .wav files from a directory and splits
  them into training and testing sets.
  """
  def __init__(self, directory: str):
    self.directory = directory

  def run(self, input_data: Optional[None], experiment_config: Dict[str, Any]) -> SplitDataResult:
    """
    Executes the data loading and splitting process.
    """
    logging.info(f"Loading and splitting data from: {self.directory}")
    
    X_train, X_test, y_train, y_test = files_train_test_split(self.directory)
    
    logging.debug("Data loaded and split successfully.")
    
    return SplitDataResult(
      X_train=X_train,
      X_test=X_test,
      y_train=y_train,
      y_test=y_test
    )