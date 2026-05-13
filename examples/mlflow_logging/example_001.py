"""
Example: log multiple trained scikit-learn models to an MLflow tracking server
via SaveModelToMLflowStep.

Run:
  1. Start the MLflow tracking server in Docker on port 5000, on a network the
     devcontainer can reach (hostname `mlflow`):

       docker run --rm -d --name mlflow --network atelierflow_default \\
         -p 5000:5000 ghcr.io/mlflow/mlflow:latest \\
         mlflow server --host 0.0.0.0 --port 5000

  2. Open this repo in VS Code and reopen in the "Atelierflow - DEV" devcontainer.

  3. From a devcontainer terminal:

       python examples/mlflow_logging/example_001.py

  4. From your host browser, inspect runs at http://localhost:5000
     (the script connects to http://mlflow:5000 from inside the devcontainer).
"""
import sys
import time
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from atelierflow.experiment import Experiment
from atelierflow.core.metric import Metric
from atelierflow.core.model import Model
from atelierflow.core.step import Step
from atelierflow.steps.common.save_model.save_to_mlflow import (
  ModelArtifact,
  SaveModelInput,
  SaveModelToMLflowStep,
)


@dataclass
class SplitDataResult:
  X_train: np.ndarray
  X_test: np.ndarray
  y_train: np.ndarray
  y_test: np.ndarray


@dataclass
class TrainedModel:
  name: str
  wrapper: Model
  train_time: float


@dataclass
class TrainingResult:
  models: List[TrainedModel]
  X_test: np.ndarray
  y_test: np.ndarray


class SklearnRandomForest(Model):
  def __init__(self, **kwargs):
    self.model = RandomForestClassifier(**kwargs)

  def fit(self, X, y=None, **kwargs):
    self.model.fit(X, y)

  def predict(self, X, **kwargs):
    return self.model.predict(X)

  def predict_proba(self, X):
    return self.model.predict_proba(X)


class AucRocMetric(Metric):
  def compute(self, y_true, y_score) -> float:
    return float(roc_auc_score(y_true, y_score))


class GenerateDataStep(Step[None, SplitDataResult]):
  def __init__(self, n_samples: int = 1000, random_state: int = 42):
    self.n_samples = n_samples
    self.random_state = random_state

  def run(self, input_data: Optional[None], experiment_config: Dict[str, Any]) -> SplitDataResult:
    logging.info(f"Generating synthetic dataset with {self.n_samples} samples...")
    X, y = make_classification(
      n_samples=self.n_samples,
      n_features=20,
      n_informative=10,
      random_state=self.random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.25, random_state=self.random_state
    )
    return SplitDataResult(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


class TrainModelsStep(Step[SplitDataResult, TrainingResult]):
  def __init__(self, models: Dict[str, Model]):
    self.models = models

  def run(self, input_data: SplitDataResult, experiment_config: Dict[str, Any]) -> TrainingResult:
    trained: List[TrainedModel] = []
    for name, model in self.models.items():
      logging.info(f"Fitting '{name}'...")
      start = time.perf_counter()
      model.fit(input_data.X_train, input_data.y_train)
      train_time = time.perf_counter() - start
      trained.append(TrainedModel(name=name, wrapper=model, train_time=train_time))
    return TrainingResult(
      models=trained,
      X_test=input_data.X_test,
      y_test=input_data.y_test,
    )


class EvaluateModelsStep(Step[TrainingResult, SaveModelInput]):
  def __init__(
    self,
    metrics: List[Metric],
    flavor,
    run_name: Optional[str] = None,
    registered_model_name_prefix: Optional[str] = None,
  ):
    self.metrics = metrics
    self.flavor = flavor
    self.run_name = run_name
    self.registered_model_name_prefix = registered_model_name_prefix

  def run(self, input_data: TrainingResult, experiment_config: Dict[str, Any]) -> SaveModelInput:
    logging.info("Scoring on test set...")
    artifacts: List[ModelArtifact] = []
    for trained in input_data.models:
      y_score = trained.wrapper.predict_proba(input_data.X_test)[:, 1]
      scores = {m.__class__.__name__: m.compute(input_data.y_test, y_score) for m in self.metrics}
      scores["train_time"] = trained.train_time
      registered_name = (
        f"{self.registered_model_name_prefix}-{trained.name}"
        if self.registered_model_name_prefix
        else None
      )
      artifacts.append(
        ModelArtifact(
          model=trained.wrapper.model,
          flavor=self.flavor,
          artifact_path=trained.name,
          metrics=scores,
          registered_model_name=registered_name,
        )
      )
    return SaveModelInput(artifacts=artifacts, run_name=self.run_name)


TRACKING_URI = "http://mlflow:5000"

experiment = Experiment(
  name="MLflow Logging Example",
  logging_level="INFO",
  tags={"project": "atelierflow-mlflow-demo", "framework": "sklearn"},
)

experiment.add_step(GenerateDataStep(n_samples=2000))
experiment.add_step(
  TrainModelsStep(
    models={
      "rf-50": SklearnRandomForest(n_estimators=50, random_state=0),
      "rf-200": SklearnRandomForest(n_estimators=200, random_state=0),
    }
  )
)
experiment.add_step(
  EvaluateModelsStep(
    metrics=[AucRocMetric()],
    flavor="sklearn",
    run_name="rf-sweep",
  )
)
experiment.add_step(SaveModelToMLflowStep(tracking_uri=TRACKING_URI))

if __name__ == "__main__":
  result = experiment.run()
  print("\n--- Final Experiment Results ---")
  for artifact in result.artifacts:
    print(f"  {artifact.artifact_path}: {artifact.metrics}")
  print(f"\nMLflow tracking server: {TRACKING_URI}")
