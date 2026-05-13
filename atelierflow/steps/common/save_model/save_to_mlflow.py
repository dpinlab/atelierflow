import importlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, Union, runtime_checkable

from atelierflow.core.step import Step

logger = logging.getLogger(__name__)


@runtime_checkable
class MLflowFlavor(Protocol):
  def log_model(self, model: Any, artifact_path: str, **kwargs: Any) -> Any: ...


FlavorName = Literal[
  "sklearn",
  "pytorch",
  "tensorflow",
  "pyfunc",
]

Flavor = Union[FlavorName, MLflowFlavor]


@dataclass
class ModelArtifact:
  model: Any
  flavor: Flavor
  artifact_path: str = "model"
  metrics: Dict[str, float] = field(default_factory=dict)
  registered_model_name: Optional[str] = None
  log_model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SaveModelInput:
  artifacts: List[ModelArtifact]
  run_metrics: Dict[str, float] = field(default_factory=dict)
  run_name: Optional[str] = None


class SaveModelToMLflowStep(Step[SaveModelInput, SaveModelInput]):
  """
  A pass-through step to log one or more trained models to MLflow.
  """

  def __init__(self, tracking_uri: Optional[str] = None, use_nested_runs: bool = True):
    self.tracking_uri = tracking_uri
    self.use_nested_runs = use_nested_runs

  @staticmethod
  def _resolve_flavor(flavor: Flavor) -> MLflowFlavor:
    if isinstance(flavor, str):
      module = importlib.import_module(f"mlflow.{flavor}")
    else:
      module = flavor
    if not hasattr(module, "log_model"):
      raise TypeError(
        f"Flavor '{getattr(module, '__name__', module)}' has no 'log_model' attribute."
      )
    return module

  def _resolve_tracking_uri(self) -> str:
    uri = self.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
      raise ValueError(
        "No MLflow tracking URI provided. Pass `tracking_uri=...` to "
        "SaveModelToMLflowStep or set the MLFLOW_TRACKING_URI environment variable."
      )
    return uri

  def _log_artifact(self, artifact: ModelArtifact) -> None:
    import mlflow

    if artifact.metrics:
      mlflow.log_metrics({k: float(v) for k, v in artifact.metrics.items()})

    flavor_module = self._resolve_flavor(artifact.flavor)
    flavor_module.log_model(
      artifact.model,
      artifact_path=artifact.artifact_path,
      registered_model_name=artifact.registered_model_name,
      **artifact.log_model_kwargs,
    )

  def run(self, input_data: SaveModelInput, experiment_config: Dict[str, Any]) -> SaveModelInput:
    import mlflow

    mlflow.set_tracking_uri(self._resolve_tracking_uri())

    experiment_name = experiment_config.get("name")
    if experiment_name:
      mlflow.set_experiment(experiment_name)

    artifacts = input_data.artifacts
    if not artifacts:
      raise ValueError("SaveModelInput.artifacts is empty; nothing to log.")

    use_nested = self.use_nested_runs and len(artifacts) > 1
    tags = experiment_config.get("tags") or {}

    with mlflow.start_run(run_name=input_data.run_name):
      if tags:
        mlflow.set_tags(tags)
      if input_data.run_metrics:
        mlflow.log_metrics({k: float(v) for k, v in input_data.run_metrics.items()})

      for artifact in artifacts:
        if use_nested:
          with mlflow.start_run(run_name=artifact.artifact_path, nested=True):
            self._log_artifact(artifact)
        else:
          self._log_artifact(artifact)

    return input_data
