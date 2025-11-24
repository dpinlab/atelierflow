from typing import Any
from dataclasses import dataclass
@dataclass
class StepResult:
  def __init__(self, **kwargs):
    self.set_attributes(kwargs)
    # self._data = {}

  def add(self, key: str, value: Any):
    self.data[key] = value

  def get(self, key: str, default: Any = None) -> Any:
    return self.data.get(key, default)