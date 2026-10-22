import threading
import queue
import time
import logging
import sys
import os
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

try:
    from ..config import config
    from ..train import USLClassifier
except ImportError:
    from config import config
    from train import USLClassifier


@dataclass
class TrainingConfig:
    actions: list
    data_path: str
    sequence_length: int
    model_params: dict
    training_params: dict
    paths: dict


class QueueHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception:
            self.handleError(record)


class TrainingWorker(threading.Thread):
    def __init__(
        self,
        config_override: Dict[str, Any],
        log_queue: queue.Queue,
        metrics_queue: queue.Queue,
        stop_event: threading.Event,
        pause_event: threading.Event,
    ):
        super().__init__(daemon=True)
        self.config_override = config_override
        self.log_queue = log_queue
        self.metrics_queue = metrics_queue
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.logger = logging.getLogger("training_worker")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = [QueueHandler(log_queue)]
        self.logger.propagate = False

    def run(self):
        try:
            self._train()
        except Exception as e:
            self.log_queue.put(f"ERROR: {str(e)}")
            self.metrics_queue.put(("error", str(e)))

    def _train(self):
        actions = self.config_override.get("actions", config.actions)
        data_path = self.config_override.get("data_path", config.data_path)
        sequence_length = self.config_override.get("sequence_length", config.sequence_length)
        model_params = self.config_override.get("model_params", config.model_params)
        training_params = self.config_override.get("training_params", config.training_params)
        paths = self.config_override.get("paths", config.paths)

        self.logger.info("Loading dataset...")
        X, y = self._load_dataset(data_path, actions, sequence_length)

        if len(X) == 0:
            self.log_queue.put("ERROR: No data loaded. Run data collection first.")
            self.metrics_queue.put(("error", "No data loaded"))
            return

        self.logger.info(f"Loaded {len(X)} sequences, {len(actions)} classes")
        self.metrics_queue.put(("data_summary", {
            "total_sequences": len(X),
            "num_classes": len(actions),
            "class_distribution": {a: int(np.sum(y == i)) for i, a in enumerate(actions)}
        }))

        val_split = training_params.get("val_split", 0.2)
        class_counts = np.bincount(y, minlength=len(actions))
        if np.any(class_counts < 2):
            missing = [actions[i] for i, count in enumerate(class_counts) if count < 2]
            message = f"At least two complete sequences are required for each class; collect more data for: {', '.join(missing)}"
            self.log_queue.put(f"ERROR: {message}")
            self.metrics_queue.put(("error", message))
            return

        validation_count = int(np.ceil(len(y) * val_split))
        training_count = len(y) - validation_count
        if validation_count < len(actions) or training_count < len(actions):
            message = "Not enough sequences for a stratified train/validation split. Collect more examples per class or increase the validation split."
            self.log_queue.put(f"ERROR: {message}")
            self.metrics_queue.put(("error", message))
            return

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=42, stratify=y
        )

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        batch_size = training_params.get("batch_size", 16)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        device_str = training_params.get("device", "cpu")
        device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
        self.logger.info(f"Using device: {device}")

        model = USLClassifier(
            input_dim=model_params.get("input_dim", 225),
            hidden_dim=model_params.get("hidden_dim", 64),
            num_layers=model_params.get("num_layers", 2),
            dropout=model_params.get("dropout", 0.2),
            num_classes=len(actions),
        ).to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=training_params.get("learning_rate", 0.001))

        best_val_loss = float("inf")
        patience_counter = 0
        total_epochs = training_params.get("epochs", 100)
        early_stopping_patience = training_params.get("early_stopping_patience", 10)
        early_stopping_min_delta = training_params.get("early_stopping_min_delta", 0.001)

        self.metrics_queue.put(("total_epochs", total_epochs))
        self.metrics_queue.put(("best_model_path", paths.get("best_model_weights", "models/usl_model_best.pt")))
        self.metrics_queue.put(("final_model_path", paths.get("model_weights", "models/usl_model.pt")))

        self.logger.info("Starting training...")
        for epoch in range(total_epochs):
            if self.stop_event.is_set():
                self.logger.info("Training stopped by user")
                self.metrics_queue.put(("stopped", epoch))
                break

            while self.pause_event.is_set() and not self.stop_event.is_set():
                self.logger.info("Training paused...")
                self.metrics_queue.put(("paused", epoch))
                time.sleep(0.5)

            if self.stop_event.is_set():
                break

            epoch_start = time.time()
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total
            epoch_time = time.time() - epoch_start

            self.logger.info(
                f"Epoch [{epoch+1}/{total_epochs}], "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Val Acc: {val_accuracy:.2f}%"
            )

            self.metrics_queue.put(("epoch_metrics", {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_acc": val_accuracy,
                "epoch_time": epoch_time,
            }))

            if avg_val_loss < best_val_loss - early_stopping_min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_path = paths.get("best_model_weights", "models/usl_model_best.pt")
                os.makedirs(os.path.dirname(best_model_path) or ".", exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
                self.logger.info(f"Saved best model to {paths.get('best_model_weights')}")
                self.metrics_queue.put(("best_model_saved", epoch + 1))
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    self.metrics_queue.put(("early_stop", epoch + 1))
                    break

        final_model_path = paths.get("model_weights", "models/usl_model.pt")
        os.makedirs(os.path.dirname(final_model_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), final_model_path)
        self.logger.info(f"Final model saved to {paths.get('model_weights')}")

        self.metrics_queue.put(("completed", {
            "final_train_loss": avg_train_loss,
            "final_val_loss": avg_val_loss,
            "final_val_acc": val_accuracy,
            "best_val_loss": best_val_loss,
            "best_epoch": epoch + 1 - patience_counter,
            "total_epochs_run": epoch + 1,
            "final_model_path": paths.get("model_weights", "models/usl_model.pt"),
            "best_model_path": paths.get("best_model_weights", "models/usl_model_best.pt"),
        }))

    def _load_dataset(self, data_path: str, actions: list, sequence_length: int):
        sequences, labels = [], []
        label_map = {label: num for num, label in enumerate(actions)}

        for action in actions:
            action_path = os.path.join(data_path, action)
            if not os.path.exists(action_path):
                self.logger.warning(f"Action path not found: {action_path}")
                continue

            sequence_dirs = sorted(
                [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))],
                key=lambda x: int(x) if x.isdigit() else 0,
            )

            for seq_dir in sequence_dirs:
                window = []
                seq_path = os.path.join(action_path, seq_dir)
                frame_files = sorted(
                    [f for f in os.listdir(seq_path) if f.endswith(".npy")],
                    key=lambda x: int(x.split(".")[0]) if x.split(".")[0].isdigit() else 0,
                )

                for frame_file in frame_files:
                    frame_path = os.path.join(seq_path, frame_file)
                    try:
                        res = np.load(frame_path)
                        window.append(res)
                    except Exception as e:
                        self.logger.error(f"Failed to load {frame_path}: {e}")
                        break

                if len(window) == sequence_length:
                    sequences.append(window)
                    labels.append(label_map[action])
                else:
                    self.logger.warning(f"Incomplete sequence {seq_dir} for {action}: {len(window)} frames")

        return np.array(sequences), np.array(labels)


def create_worker(
    config_override: Dict[str, Any],
    log_queue: queue.Queue,
    metrics_queue: queue.Queue,
    stop_event: threading.Event,
    pause_event: threading.Event,
) -> TrainingWorker:
    return TrainingWorker(config_override, log_queue, metrics_queue, stop_event, pause_event)