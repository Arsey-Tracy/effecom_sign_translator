# USL Translator - Ugandan Sign Language Live Translator

A real-time sign language recognition system for Ugandan Sign Language (USL) using MediaPipe for landmark detection and PyTorch LSTM for sequence classification.

## Features

- **Live Inference**: Real-time sign recognition from webcam
- **Data Collection**: Guided data collection for custom signs
- **Model Training**: LSTM-based classifier with validation and early stopping
- **CPU Optimized**: Runs efficiently on CPU-only environments
- **Configurable**: Centralized YAML configuration
- **CLI Interface**: Single entry point for all operations

## Supported Signs

- `oli_otya` - Hello
- `oli_bulungi` - How are you
- `webale` - Thank you

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Collect Training Data

```bash
python -m src.run collect
```

Follow on-screen prompts to record 30 sequences of 30 frames for each sign.

### 2. Train the Model

```bash
python -m src.run train
```

Trains an LSTM classifier with validation split and early stopping.

### 3. Run Live Inference

```bash
python -m src.run infer
```

Opens webcam and displays recognized signs in real-time.

## Configuration

All settings are in `config.yaml`:

```yaml
actions:
  - "oli_otya"
  - "oli_bulungi"
  - "webale"

data:
  processed_path: "data/processed_npy"
  no_sequences: 30
  sequence_length: 30

model:
  input_dim: 225
  hidden_dim: 64
  num_layers: 2
  dropout: 0.2

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001
  val_split: 0.2
  early_stopping_patience: 10
  device: "cpu"

inference:
  confidence_threshold: 0.75
  buffer_size: 30
  camera_index: 0
```

## CLI Options

```bash
# Override config path
python -m src.run train --config custom_config.yaml

# Force CPU/GPU
python -m src.run infer --device cpu

# Custom training epochs
python -m src.run train --epochs 50

# Custom confidence threshold
python -m src.run infer --confidence 0.8
```

## Project Structure

```bash
usl_translator/
├── config.yaml              # Centralized configuration
├── requirements.txt         # Python dependencies
├── src/
│   ├── config.py           # Config loader (singleton)
│   ├── extractor.py        # MediaPipe feature extraction
│   ├── train.py            # Model training with validation
│   ├── main.py             # Live inference
│   ├── builder.py          # Data collection
│   ├── run.py              # CLI entry point
│   └── models/             # MediaPipe models & trained weights
├── tests/
│   ├── test_core.py        # Unit tests
│   └── conftest.py         # Pytest fixtures
└── data/
    └── processed_npy/      # Training data (generated)
```

## Model Architecture

- **Input**: 30 frames × 225 features (33 pose × 3 + 2×21 hand × 3)
- **LSTM**: 2 layers, 64 hidden units, 20% dropout
- **Output**: 3 classes (softmax)
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam (lr=0.001)

## CPU Optimizations

- Tensor reuse to avoid repeated allocations
- Batch size configurable for memory management
- `torch.no_grad()` for inference
- `map_location=device` for model loading
- No CUDA dependencies

## Testing

```bash
pytest tests/ -v
```

## Requirements

- Python 3.8+
- MediaPipe 0.10+
- PyTorch 2.3+
- OpenCV 4.10+
- NumPy 1.26+
- PyYAML 6.0+

## MediaPipe Models

Download the following models to `src/models/`:

- `hand_landmarker.task`
- `pose_landmarker.task`
- `holistic_landmarker.task` (optional)

Available from: <https://developers.google.com/mediapipe/solutions/vision>

## License

MIT License
