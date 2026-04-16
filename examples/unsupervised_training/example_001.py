import sys
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from mtsa.models import IForest
from mtsa.utils.utils import files_train_test_split
from atelierflow.experiment import Experiment
from atelierflow.core.metric import Metric
from atelierflow.core.model import Model
from atelierflow.core.step import Step
from atelierflow.steps.common.save_data.save_to_avro import SaveToAvroStep

@dataclass
class SplitDataResult:
  X_train: np.ndarray
  X_test: np.ndarray
  y_train: np.ndarray
  y_test: np.ndarray

@dataclass
class TrainingResult:
  model: Model
  X_test: np.ndarray
  y_test: np.ndarray

@dataclass
class EvaluationResult:
  evaluation_scores: Dict[str, float]

class MyIForest(Model):
  def __init__(self, **kwargs):
    self.model = IForest(**kwargs)

  def fit(self, X):
    self.model.fit(X)

  def predict(self, X):
    return self.model.predict(X)

  def score_samples(self, X):
    return self.model.score_samples(X)

class AucRocMetric(Metric):
  def __init__(self, name):
    self.name = name

  def compute(self, y_true, y_pred) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_pred))

class LoadAndSplitDataStep(Step[None, SplitDataResult]):
  def __init__(self, directory: str):
    self.directory = directory

  def run(self, input_data: Optional[None], experiment_config: Dict[str, Any]) -> SplitDataResult:
    logging.info(f"Loading and splitting data from: {self.directory}")
    X_train, X_test, y_train, y_test = files_train_test_split(self.directory)
    return SplitDataResult(
      X_train=X_train,
      X_test=X_test,
      y_train=y_train,
      y_test=y_test
    )

class TrainModelStep(Step[SplitDataResult, TrainingResult]):
  def __init__(self, model: Model):
    self.model = model

  def run(self, input_data: SplitDataResult, experiment_config: Dict[str, Any]) -> TrainingResult:
    logging.info(f"Fitting model '{self.model.__class__.__name__}'...")
    self.model.fit(input_data.X_train)
    return TrainingResult(model=self.model, X_test=input_data.X_test, y_test=input_data.y_test)

class EvaluateModelStep(Step[TrainingResult, EvaluationResult]):
  def __init__(self, metrics: List[Metric]):
    self.metrics = metrics

  def run(self, input_data: TrainingResult, experiment_config: Dict[str, Any]) -> EvaluationResult:
    logging.info("Making predictions on the test set...")
    y_pred = input_data.model.score_samples(input_data.X_test)
    scores = {}
    for metric in self.metrics:
      score = metric.compute(input_data.y_test, y_pred)
      scores[metric.__class__.__name__] = score
    return EvaluationResult(evaluation_scores=scores)

DATA_DIRECTORY = "/home/celin/Desktop/codes/lab/atelierflow/examples/sample_data/machine_type_1/id_00"
OUTPUT_FILE = "./anomaly_detection_results.avro"

model_component = MyIForest()
metric_component = AucRocMetric(name="AUC-ROC")

iforest_experiment = Experiment(name="Isolation Forest Anomaly Detection", logging_level="INFO")

iforest_experiment.add_step(LoadAndSplitDataStep(directory=DATA_DIRECTORY))
iforest_experiment.add_step(TrainModelStep(model=model_component))
iforest_experiment.add_step(EvaluateModelStep(metrics=[metric_component]))

scores_schema = {
    'doc': 'Schema for storing model evaluation scores.',
    'name': 'EvaluationScores',
    'type': 'record',
    'fields': [
        {'name': 'AucRocMetric', 'type': ['null', 'double'], 'default': None}
    ]
}

iforest_experiment.add_step(
    SaveToAvroStep(
        output_path=OUTPUT_FILE,
        data_key='evaluation_scores',
        schema=scores_schema
    )
)

final_results = iforest_experiment.run()

print("\n--- Final Experiment Results ---")
print(final_results.evaluation_scores)