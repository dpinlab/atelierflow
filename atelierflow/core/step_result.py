from typing import Any
from dataclasses import dataclass
@dataclass
class StepResult:
  def __init__(self, **kwargs):
    self.set_attributes(kwargs)

  def set_attributes(self, attributes: dict[str, Any]):
    for key, value in attributes.items():
      setattr(self, key, value)