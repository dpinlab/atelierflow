import logging
import os
from typing import Any, Dict, Optional, TypeVar, Generic, List, Union
import pandas as pd
from fastavro import writer, parse_schema

from atelierflow.core.step import Step

logger = logging.getLogger(__name__)

T = TypeVar("T")

class SaveToAvroStep(Step[T, T]):
  """
  A generic step to save data into an Avro file.
  It is a pass-through step: it returns exactly what it receives.
  """
  def __init__(
      self,
      output_path: str,
      data_key: str,
      schema: Optional[Dict[str, Any]] = None,
      append: bool = True
  ):
    self.output_path = output_path
    self.data_key = data_key
    self.append = append
    self.parsed_schema = parse_schema(schema) if schema else None

    if self.append and not self.parsed_schema:
      raise ValueError("A 'schema' must be provided when using append mode.")

  def run(self, input_data: T, experiment_config: Dict[str, Any]) -> T:
    if not input_data:
      raise ValueError("SaveToAvroStep requires input data.")

    data_to_save = getattr(input_data, self.data_key, None)
    
    if data_to_save is None:
      raise ValueError(f"Data key '{self.data_key}' not found in {type(input_data).__name__}")

    records = self._get_records(data_to_save)

    if not records:
      logger.warning(f"No records found for key '{self.data_key}'.")
      return input_data

    schema_to_use = self.parsed_schema
    file_exists = os.path.exists(self.output_path)
    mode = 'a+b' if self.append and file_exists else 'wb'

    if not schema_to_use:
      schema_to_use = self._infer_schema(records)

    with open(self.output_path, mode) as out:
      writer(out, schema_to_use, records)

    return input_data

  def _get_records(self, data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, pd.DataFrame):
      return data.to_dict('records')
    if isinstance(data, list) and all(isinstance(i, dict) for i in data):
      return data
    if isinstance(data, dict):
      return [data]
    raise TypeError(f"Unsupported data type for Avro save: {type(data)}")

  def _infer_schema(self, records: List[Dict[str, Any]]) -> Any:
    inferred = {
      'doc': 'Inferred by AtelierFlow', 
      'name': 'InferredRecord', 
      'type': 'record',
      'fields': [{'name': k, 'type': ['null', self._infer_avro_type(v)]} for k, v in records[0].items()],
    }
    return parse_schema(inferred)

  def _infer_avro_type(self, value: Any) -> str:
    if isinstance(value, int): return 'long'
    if isinstance(value, float): return 'double'
    if isinstance(value, bool): return 'boolean'
    return 'string'