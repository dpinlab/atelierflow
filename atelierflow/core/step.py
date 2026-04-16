from abc import ABC, abstractmethod
from typing import Dict, Any, Generic, TypeVar

T_in = TypeVar('T_in')
T_out = TypeVar('T_out')

class Step(ABC, Generic[T_in, T_out]):
  """
  Defines the contract for any executable step in an AtelierFlow pipeline.

  Any class that inherits from Step must implement the `run` method.
  The constructor (__init__) of the subclass is where dependencies
  (such as models, metrics, or configurations) should be injected.
  """

  @abstractmethod
  def run(self, input_data: T_in, experiment_config: Dict[str, Any]) -> T_out:
    """
    Executes the main logic of the step.

    :param input_data: The result object from the previous step.
                        Will be `None` if this is the first step in the pipeline.
    :return: A new StepResult object containing the output of this step.
    """
    raise NotImplementedError("Subclasses de Step devem implementar o método 'run'.")
