from dataclasses import dataclass
import logging
from typing import Any, Dict, List
import copy
import numpy as np
from sklearn.model_selection import KFold
import gc 

from atelierflow.core.model import Model
from atelierflow.core.metric import Metric
from atelierflow.core.step import Step
from atelierflow.steps.mtsa.preprocessing.load_and_split_data import SplitDataResult

logger = logging.getLogger(__name__)


@dataclass
class MetricCVResult:
    scores: List[float]
    mean: float
    std: float
    artifacts: List[Dict[str, Any]]

@dataclass
class CrossValidationResult:
    cv_metrics: Dict[str, MetricCVResult]
    X_test: np.ndarray
    y_test: np.ndarray

class UnsupervisedCrossValidationStep(Step[SplitDataResult, CrossValidationResult]):
  """
  Performs cross-validation for an unsupervised model.

  This step trains the model on different subsets (folds) of the training data
  and evaluates the performance of each trained model on a fixed test set.
  """
  def __init__(self, model: Model, metrics: List[Metric], n_splits=5):
    super().__init__()
    self.model = model
    self.metrics = metrics
    self.n_splits = n_splits

  def run(self, input_data: SplitDataResult, experiment_config: Dict[str, Any]) -> CrossValidationResult:
    X_train = input_data.X_train
    y_train = input_data.y_train 
    X_test = input_data.X_test
    y_test = input_data.y_test

    kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
    
    logger.info(f"Starting {self.n_splits}-fold cross-validation for unsupervised model.")
    cv_scores = {metric.name: [] for metric in self.metrics}
    cv_artifacts = {metric.name: [] for metric in self.metrics}

    for fold, (train_index, _) in enumerate(kf.split(X_train, y_train)):
      logger.info(f"--- Fold {fold + 1}/{self.n_splits} ---")
      
      x_train_fold, _ = X_train[train_index], y_train[train_index]

      model_for_fold = copy.deepcopy(self.model)
      
      logger.debug(f"Training model for fold {fold + 1}...")
      model_for_fold.fit(x_train_fold) 

      logger.debug(f"Evaluating model on the fixed test set...")
      
      predictions_or_scores = model_for_fold.predict(X_test)
      
      for metric in self.metrics:
        result = metric.compute(y_true=y_test, y_score=predictions_or_scores)
        
        score = None
        artifacts = {} 

        if isinstance(result, tuple):
          score, artifacts = result
        else:
          score = result
        # ------------------------------------

        cv_scores[metric.name].append(score)
        cv_artifacts[metric.name].append(artifacts) 
        logger.info(f"{metric.name} for fold {fold + 1}: {score:.4f}")
      
      del model_for_fold
      gc.collect()

    logger.info("Cross-validation evaluation complete.")
    
    final_metrics = {}
    for name, scores in cv_scores.items():
      final_metrics[name] = {
        'scores': scores,
        'mean': np.mean(scores),
        'std': np.std(scores),
        'artifacts': cv_artifacts[name]
      }
      logger.info(f"Final {name}: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

    return CrossValidationResult(
      cv_metrics=final_metrics,
      X_test=X_test,
      y_test=y_test
    )