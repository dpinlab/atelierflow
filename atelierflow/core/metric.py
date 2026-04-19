from typing import Protocol, runtime_checkable

@runtime_checkable
class Metric(Protocol):
  """
  Structural interface for performance evaluation metrics.
  
  Any class implementing the `compute` method is considered a valid Metric,
  allowing for flexible integration of custom scoring logic.
  """
  name: str
  
  def compute(self, **kwargs):
    """
    Computes the metric.
    """
    ...
