import logging
from mtsa.utils import files_train_test_split
from typing import Any, Dict, Optional
from atelierflow.core.step import Step
from atelierflow.core.step_result import StepResult

logger = logging.getLogger(__name__)


class LoadAndSplitDataStep(Step):
  """
  A preprocessing step that loads .wav files from a directory and splits
  them into training and testing sets.
  """
  def __init__(self, directory: str):
    """
    Initializes the data loading and splitting step.

    Args:
      directory (str): The full path to the root directory containing the .wav files.
        This directory must contain two subdirectories: 'normal' and 'abnormal'.
        The 'normal' subdirectory should contain all normal audio files, and the
        'abnormal' subdirectory should contain all abnormal audio files.
    """
    self.directory = directory

  def run(self, input_data: Optional[StepResult], experiment_config: Dict[str, Any]) -> StepResult:
    """
    Executes the data loading and splitting process.

    Args:
      input_data (Optional[StepResult]): Input data from a previous step (not used here).
      experiment_config (Dict[str, Any]): The experiment's configuration dictionary.

    Returns:
      StepResult: An object containing the training and testing data splits
                  ('X_train', 'X_test', 'y_train', 'y_test').
    """
    logging.info(f"Loading and splitting data from: {self.directory}")
    X_train, X_test, y_train, y_test = files_train_test_split(self.directory)
    logging.debug("Data loaded and split successfully.")
    result = StepResult()
    result.add('X_train', X_train)
    result.add('X_test', X_test)
    result.add('y_train', y_train)
    result.add('y_test', y_test)
    return result