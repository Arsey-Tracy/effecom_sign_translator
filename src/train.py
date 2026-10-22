import os
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Callable, Optional, Dict, Any, List
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

try:
    from .config import config
except ImportError:
    from config import config


logging.basicConfig(
    level=getattr(logging, config.get("logging.level", "INFO")),
    format=config.get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
)
logger = logging.getLogger(__name__)


class USLClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 225,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 3,
    ):
        super(USLClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def load_dataset(data_path: str, actions: list, sequence_length: int):
    sequences, labels = [], []
    label_map = {label: num for num, label in enumerate(actions)}

    for action in actions:
        action_path = os.path.join(data_path, action)
        if not os.path.exists(action_path):
            logger.warning(f"Action path not found: {action_path}")
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
                    logger.error(f"Failed to load {frame_path}: {e}")
                    break

            if len(window) == sequence_length:
                sequences.append(window)
                labels.append(label_map[action])
            else:
                logger.warning(f"Incomplete sequence {seq_dir} for {action}: {len(window)} frames")

    return np.array(sequences), np.array(labels)


def train_model(
    config_override: Optional[Dict[str, Any]] = None,
    on_epoch_end: Optional[Callable[[int, Dict[str, float]], None]] = None,
    on_log: Optional[Callable[[str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    should_pause: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Train the USL classifier with optional callbacks for progress monitoring.
    
    Args:
        config_override: Optional dict to override config values
        on_epoch_end: Callback(epoch, metrics_dict) called after each epoch
        on_log: Callback(message) for log messages
        should_stop: Callback returning True if training should stop
        should_pause: Callback returning True if training should pause
    
    Returns:
        Dict with final training results
    """
    cfg = config_override or {}
    
    actions = cfg.get("actions", config.actions)
    data_path = cfg.get("data_path", config.data_path)
    sequence_length = cfg.get("sequence_length", config.sequence_length)
    model_params = cfg.get("model_params", config.model_params)
    training_params = cfg.get("training_params", config.training_params)
    paths = cfg.get("paths", config.paths)

    def log(msg: str):
        if on_log:
            on_log(msg)
        else:
            logger.info(msg)

    log("Loading dataset...")
    X, y = load_dataset(data_path, actions, sequence_length)

    if len(X) == 0:
        log("No data loaded. Run data collection first.")
        return {"error": "No data loaded"}

    log(f"Loaded {len(X)} sequences, {len(actions)} classes")

    val_split = training_params.get("val_split", 0.2)
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
    log(f"Using device: {device}")

    model = USLClassifier(
        input_dim=model_params.get("input_dim", 225),
        hidden_dim=model_params.get("hidden_dim", 64),
        num_layers=model_params.get("num_layers", 2),
        dropout=model_params.get("dropout", 0.2),
        num_classes=len(actions),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_params.get("learning_rate", 0.001))

    best_val_loss = float("inf")
    patience_counter = 0
    total_epochs = training_params.get("epochs", 100)
    early_stopping_patience = training_params.get("early_stopping_patience", 10)
    early_stopping_min_delta = training_params.get("early_stopping_min_delta", 0.001)

    log("Starting training...")
    final_metrics = {}

    for epoch in range(total_epochs):
        if should_stop and should_stop():
            log("Training stopped by user")
            final_metrics["stopped"] = True
            final_metrics["epoch"] = epoch
            break

        while should_pause and should_pause() and not (should_stop and should_stop()):
            log("Training paused...")
            time.sleep(0.5)

        if should_stop and should_stop():
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

        log(
            f"Epoch [{epoch+1}/{total_epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Acc: {val_accuracy:.2f}%"
        )

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_acc": val_accuracy,
            "epoch_time": epoch_time,
        }

        if on_epoch_end:
            on_epoch_end(epoch + 1, epoch_metrics)

        if avg_val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(paths.get("best_model_weights", "models/usl_model_best.pt")), exist_ok=True)
            torch.save(model.state_dict(), paths.get("best_model_weights", "models/usl_model_best.pt"))
            log(f"Saved best model to {paths.get('best_model_weights')}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                log(f"Early stopping triggered after {epoch+1} epochs")
                final_metrics["early_stop"] = True
                final_metrics["early_stop_epoch"] = epoch + 1
                break

    os.makedirs(os.path.dirname(paths.get("model_weights", "models/usl_model.pt")), exist_ok=True)
    torch.save(model.state_dict(), paths.get("model_weights", "models/usl_model.pt"))
    log(f"Final model saved to {paths.get('model_weights')}")

    final_metrics.update({
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
        "final_val_acc": val_accuracy,
        "best_val_loss": best_val_loss,
        "best_epoch": epoch + 1 - patience_counter,
        "total_epochs_run": epoch + 1,
        "model_path": paths.get("model_weights"),
        "best_model_path": paths.get("best_model_weights"),
    })

    return final_metrics


if __name__ == "__main__":
    train_model()