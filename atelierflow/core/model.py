from typing import Protocol, runtime_checkable

@runtime_checkable
class Model(Protocol):
  """
  Defines the structural contract for any machine learning model.

  This is a pure "component" that decouples the ML algorithm from the 
  pipeline logic, allowing for seamless integration of any class that 
  implements 'fit' and 'predict'.
  """

  def fit(self, X, y=None, **kwargs):
    """
    Trains the model with the provided data.
    """
    ...
  def predict(self, X, **kwargs):
    """
    Performs predictions on new data after the model has been trained.
    """
    ...