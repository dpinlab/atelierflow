import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# For this example, we use the mtsa package.

from mtsa.models import IForest

from atelierflow.experiment import Experiment
from atelierflow.core.metric import Metric
from atelierflow.core.model import Model
from atelierflow.steps.common.save_data.save_to_avro import SaveToAvroStep
from atelierflow.steps.mtsa.preprocessing.load_and_split_data import LoadAndSplitDataStep
from atelierflow.steps.sklearn.validation.cross_validation import UnsupervisedCrossValidationStep

# --- Components ---
class MyIForest(Model):
  def __init__(self, **kwargs):
    self.model = IForest(**kwargs)

  def fit(self, X, y=None):
    self.model.fit(X, y)

  def predict(self, X):
    return self.model.predict(X)

  def score_samples(self, X):
    return self.model.score_samples(X)
  
class AucRocMetric(Metric):
  def __init__(self, name):
    self.name=name

  def compute(self, y_true, y_pred) -> float:
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    
    roc_auc = auc(fpr, tpr)
    
    artifacts = {
      'fpr': fpr.tolist(),
      'tpr': tpr.tolist()
    }
    
    return roc_auc, artifacts
  
# --- 1. Configuration ---
DATA_DIRECTORY = "/data/marcelo/atelierflow/examples/sample_data/machine_type_1/id_00"
OUTPUT_FILE = "./anomaly_detection_results.avro"

model_component = MyIForest()  
metric_component = AucRocMetric(name="AUC-ROC")

# --- 2. Pipeline Assembly ---
iforest_experiment = Experiment(name="Isolation Forest Anomaly Detection", logging_level="INFO")

iforest_experiment.add_step(LoadAndSplitDataStep(directory=DATA_DIRECTORY))
iforest_experiment.add_step(UnsupervisedCrossValidationStep(model=model_component, metrics=[metric_component]))

scores_schema = {
  "namespace": "atelierflow.experiment",
  "type": "map",
  "doc": "A map from a metric name (e.g., 'AUC-ROC') to its detailed results.",
  "values": {
    "name": "MetricResult",
    "type": "record",
    "doc": "Aggregated results for a single performance metric across all folds.",
    "fields": [
      {
        "name": "scores",
        "type": { "type": "array", "items": "double" },
        "doc": "List of the numerical scores for each fold."
      },
      {
        "name": "mean",
        "type": "double",
        "doc": "The mean score across all folds."
      },
      {
        "name": "std",
        "type": "double",
        "doc": "The standard deviation of scores across all folds."
      },
      {
        "name": "artifacts",
        "type": {
          "type": "array",
          "items": {
            "type": "map",
            "values": [
              "null",
              "string",
              "long",
              "double",
              "boolean",
              { "type": "array", "items": "double" }
            ]
          },
          "doc": "A list of artifact dictionaries, one for each fold."
        }
      }
    ]
  }
}

iforest_experiment.add_step(
  SaveToAvroStep(
    output_path=OUTPUT_FILE,
    data_key='cv_metrics',
    schema=scores_schema
  )
)
# --- 3. Execution ---
final_results = iforest_experiment.run()

print("\n--- Final Experiment Results ---")
scores = final_results.cv_metrics
print(scores)