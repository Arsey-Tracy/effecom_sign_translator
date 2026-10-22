import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    _instance: Optional["Config"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._config:
            self.load()

    def load(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        config_path = Path(config_path).resolve()

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    @property
    def actions(self):
        return self.get("actions", [])

    @property
    def data_path(self):
        return self.get("data.processed_path", "data/processed_npy")

    @property
    def no_sequences(self):
        return self.get("data.no_sequences", 30)

    @property
    def sequence_length(self):
        return self.get("data.sequence_length", 30)

    @property
    def model_params(self):
        return {
            "input_dim": self.get("model.input_dim", 225),
            "hidden_dim": self.get("model.hidden_dim", 64),
            "num_layers": self.get("model.num_layers", 2),
            "dropout": self.get("model.dropout", 0.2),
        }

    @property
    def training_params(self):
        return {
            "epochs": self.get("training.epochs", 100),
            "batch_size": self.get("training.batch_size", 16),
            "learning_rate": self.get("training.learning_rate", 0.001),
            "val_split": self.get("training.val_split", 0.2),
            "early_stopping_patience": self.get("training.early_stopping_patience", 10),
            "early_stopping_min_delta": self.get("training.early_stopping_min_delta", 0.001),
            "device": self.get("training.device", "cpu"),
        }

    @property
    def inference_params(self):
        return {
            "confidence_threshold": self.get("inference.confidence_threshold", 0.75),
            "buffer_size": self.get("inference.buffer_size", 30),
            "camera_index": self.get("inference.camera_index", 0),
            "display_width": self.get("inference.display_width", 640),
            "display_height": self.get("inference.display_height", 480),
        }

    @property
    def paths(self):
        return {
            "hand_model": self.get("paths.hand_model", "models/hand_landmarker.task"),
            "pose_model": self.get("paths.pose_model", "models/pose_landmarker.task"),
            "holistic_model": self.get("paths.holistic_model", "models/holistic_landmarker.task"),
            "model_weights": self.get("paths.model_weights", "models/usl_model.pt"),
            "best_model_weights": self.get("paths.best_model_weights", "models/usl_model_best.pt"),
        }


config = Config()