# USL Translator

A webcam-based Ugandan Sign Language (USL) recognition project. MediaPipe extracts pose and hand landmarks from video frames, and a PyTorch LSTM classifies each sequence as one of the configured signs.

## Features

- Record labeled sign sequences from a webcam.
- Train an LSTM classifier with a validation split and early stopping.
- Run live sign recognition from a webcam.
- Use the command-line interface or the Streamlit training dashboard.
- Adjust data, model, training, and inference settings in `config.yaml`.

The default classes are:

| Class | Meaning |
| --- | --- |
| `oli_otya` | Hello |
| `oli_bulungi` | How are you |
| `webale` | Thank you |

## Requirements

- Python and pip, using a version supported by the pinned packages in `requirements.txt`.
- A webcam for data collection and live inference.
- The MediaPipe task model files configured under `paths` in `config.yaml`.

Install the project dependencies from the repository root:

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

```bash
# macOS / Linux
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Streamlit dashboard

The dashboard provides data collection and interactive model training controls. Start it from the `src` directory:

```bash
cd src
streamlit run streamlit_app.py
```

Streamlit prints a local URL to open in your browser. Use the sidebar to configure sign classes and training settings, and the page controls to collect data or start, pause, and stop training.

## Command-line workflows

Run these commands from the `src` directory. This is important because data and model paths in `config.yaml` are relative to the current working directory.

```bash
cd src

# Record sequences for the configured sign classes
python run.py collect

# Train a model from the recorded sequences
python run.py train

# Run webcam inference using the trained model
python run.py infer
```

Data collection records the configured number of sequences for each class, with the configured number of frames per sequence. Press `q` in the OpenCV window to stop collection. Training requires valid sequences for every configured class so a stratified training/validation split can be created. Inference requires trained model weights.

### CLI options

```bash
# Load another configuration file (path is relative to the current directory)
python run.py train --config ../custom_config.yaml

# Override the device or epoch count
python run.py train --device cpu --epochs 50

# Set the inference confidence threshold
python run.py infer --confidence 0.8
```

The available options are `--config`, `--device` (`cpu` or `cuda`), `--epochs`, and `--confidence`.

## Configuration

`config.yaml` is at the repository root and is loaded by default. The main settings are:

| Section | Settings |
| --- | --- |
| `actions` | Sign class names used for data folders, training labels, and predictions |
| `data` | Processed-data directory, sequences per class, and frames per sequence |
| `model` | Input size, LSTM hidden size/layers, and dropout |
| `training` | Epochs, batch size, learning rate, validation split, early stopping, and device |
| `inference` | Confidence threshold, frame buffer, camera index, and display dimensions |
| `paths` | MediaPipe task assets and trained model weight files |
| `logging` | Log level, message format, and log file |

Configured data and model paths are relative to the directory from which the command or Streamlit app is launched. The commands above run from `src`, where the default paths point to `src/data` and `src/models`.

## Data and model files

The data collector writes each frame as a NumPy `.npy` file in this layout:

```text
src/data/processed_npy/<sign>/<sequence>/<frame>.npy
```

Each frame contains 225 values: 33 pose landmarks and 21 landmarks for each of two hands, with three coordinates per landmark. A complete training example consists of `data.sequence_length` frames.

The configured hand and pose MediaPipe task files (`hand_landmarker.task` and `pose_landmarker.task`) must exist at the paths in `config.yaml`. If these files are missing, download compatible task models from the [MediaPipe documentation](https://developers.google.com/mediapipe/solutions/vision). Training writes the best and final PyTorch weights to the configured model paths. The default inference workflow loads the final weights.

## Model

The default classifier accepts a sequence of 30 frames, each with 225 features. It uses a two-layer LSTM with 64 hidden units and 20% dropout, followed by a linear layer that produces one score per configured class. Training uses cross-entropy loss and the Adam optimizer; early stopping monitors validation loss.

## Project layout

```text
usl_translator/
├── config.yaml
├── requirements.txt
├── LICENSE.txt
├── src/
│   ├── builder.py                 # Webcam data collection
│   ├── config.py                  # YAML configuration loader
│   ├── extractor.py               # MediaPipe landmark extraction
│   ├── main.py                    # Live inference
│   ├── run.py                     # CLI entry point
│   ├── train.py                   # Model and training routine
│   ├── streamlit_app.py           # Streamlit dashboard entry point
│   ├── streamlit_pages/
│   │   └── train.py               # Training dashboard UI
│   ├── streamlit_utils/            # Dashboard state, worker, and settings
│   ├── models/                    # MediaPipe assets and model weights
│   └── data/processed_npy/        # Collected training data
└── tests/
    ├── conftest.py
    └── test_core.py
```

## Tests

From the repository root, run:

```bash
python -m pytest tests/ -v
```
