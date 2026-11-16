
from dataclasses import dataclass
from pathlib import Path

from src import config as C


@dataclass(frozen=True)
class TaskSpec:
    key: str
    title: str
    labels: tuple
    models_dir: Path
    reports_dir: Path
    model_keys: tuple
    blurb: str
    weak_labels: bool = False

    @property
    def n_classes(self) -> int:
        return len(self.labels)

    @property
    def label_to_id(self) -> dict:
        return {l: i for i, l in enumerate(self.labels)}

    @property
    def evaluation_path(self) -> Path:
        return self.reports_dir / "evaluation.json"

    @property
    def predictions_path(self) -> Path:
        return self.reports_dir / "test_predictions.csv"


THREE_CLASS = TaskSpec(
    key="three_class",
    title="Positive / Neutral / Negative",
    labels=tuple(C.THREE_LABELS),
    models_dir=C.THREE_MODELS_DIR,
    reports_dir=C.THREE_REPORTS_DIR,
    model_keys=("naive_bayes", "logistic_regression", "linear_svm", "mlp", "textcnn", "bilstm"),
    blurb=("English Amazon fine-food and women's-clothing reviews with labels derived from star ratings "
           "(1-2 = Negative, 3 = Neutral, 4-5 = Positive)."),
    weak_labels=True,
)

TASKS = {THREE_CLASS.key: THREE_CLASS}
TASK = THREE_CLASS


def get_task(key: str = "three_class") -> TaskSpec:
    if key not in TASKS:
        raise ValueError(f"Unknown task '{key}'. SentimentIQ supports only: {list(TASKS)}.")
    return TASKS[key]
