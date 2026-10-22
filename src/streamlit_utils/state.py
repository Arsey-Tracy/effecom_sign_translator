import streamlit as st
import time
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum


class TrainingStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float
    epoch_time: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrainingState:
    status: TrainingStatus = TrainingStatus.IDLE
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    current_epoch: int = 0
    total_epochs: int = 0
    metrics_history: List[EpochMetrics] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    patience_counter: int = 0
    start_time: Optional[float] = None
    error_message: Optional[str] = None
    log_lines: List[str] = field(default_factory=list)
    data_summary: Optional[Dict[str, Any]] = None
    final_model_path: Optional[str] = None
    best_model_path: Optional[str] = None
    final_results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "config_snapshot": self.config_snapshot,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "metrics_history": [asdict(m) for m in self.metrics_history],
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "patience_counter": self.patience_counter,
            "start_time": self.start_time,
            "error_message": self.error_message,
            "log_lines": self.log_lines,
            "data_summary": self.data_summary,
            "final_model_path": self.final_model_path,
            "best_model_path": self.best_model_path,
            "final_results": self.final_results,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingState":
        state = cls()
        state.status = TrainingStatus(data.get("status", "idle"))
        state.config_snapshot = data.get("config_snapshot", {})
        state.current_epoch = data.get("current_epoch", 0)
        state.total_epochs = data.get("total_epochs", 0)
        state.metrics_history = [
            EpochMetrics(**m) for m in data.get("metrics_history", [])
        ]
        state.best_val_loss = data.get("best_val_loss", float("inf"))
        state.best_epoch = data.get("best_epoch", 0)
        state.patience_counter = data.get("patience_counter", 0)
        state.start_time = data.get("start_time")
        state.error_message = data.get("error_message")
        state.log_lines = data.get("log_lines", [])
        state.data_summary = data.get("data_summary")
        state.final_model_path = data.get("final_model_path")
        state.best_model_path = data.get("best_model_path")
        state.final_results = data.get("final_results", {})
        return state


def init_training_state() -> TrainingState:
    if "training_state" not in st.session_state:
        st.session_state.training_state = TrainingState()
    return st.session_state.training_state


def get_training_state() -> TrainingState:
    return st.session_state.get("training_state", TrainingState())


def reset_training_state():
    st.session_state.training_state = TrainingState()


def update_training_state(**kwargs):
    state = get_training_state()
    for key, value in kwargs.items():
        if hasattr(state, key):
            setattr(state, key, value)


def add_log_line(line: str):
    state = get_training_state()
    state.log_lines.append(f"[{time.strftime('%H:%M:%S')}] {line}")
    if len(state.log_lines) > 1000:
        state.log_lines = state.log_lines[-1000:]


def add_epoch_metrics(metrics: EpochMetrics):
    state = get_training_state()
    state.metrics_history.append(metrics)
    state.current_epoch = metrics.epoch
    if metrics.val_loss < state.best_val_loss:
        state.best_val_loss = metrics.val_loss
        state.best_epoch = metrics.epoch
        state.patience_counter = 0
    else:
        state.patience_counter += 1


def get_metrics_for_chart() -> Dict[str, List[float]]:
    state = get_training_state()
    if not state.metrics_history:
        return {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}
    return {
        "epoch": [m.epoch for m in state.metrics_history],
        "train_loss": [m.train_loss for m in state.metrics_history],
        "val_loss": [m.val_loss for m in state.metrics_history],
        "val_acc": [m.val_acc for m in state.metrics_history],
    }


def get_elapsed_time() -> float:
    state = get_training_state()
    if state.start_time is None:
        return 0.0
    return time.time() - state.start_time


def get_eta() -> Optional[float]:
    state = get_training_state()
    if state.current_epoch <= 0 or state.total_epochs <= 0:
        return None
    elapsed = get_elapsed_time()
    avg_epoch_time = elapsed / state.current_epoch
    remaining_epochs = state.total_epochs - state.current_epoch
    return avg_epoch_time * remaining_epochs


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"