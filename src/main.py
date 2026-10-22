import cv2
import torch
import numpy as np
import logging
from collections import deque

try:
    from .config import config
except ImportError:
    from config import config
try:
    from .extractor import USLFeatureExtractor
    from .train import USLClassifier
except ImportError:
    from extractor import USLFeatureExtractor
    from train import USLClassifier


logger = logging.getLogger(__name__)


def run_inference():
    actions = config.actions
    inf_params = config.inference_params
    paths = config.paths
    model_params = config.model_params
    train_params = config.training_params

    device = torch.device(train_params["device"])
    logger.info(f"Using device: {device}")

    try:
        model = USLClassifier(
            input_dim=model_params["input_dim"],
            hidden_dim=model_params["hidden_dim"],
            num_layers=model_params["num_layers"],
            dropout=model_params["dropout"],
            num_classes=len(actions),
        ).to(device)

        model.load_state_dict(torch.load(paths["model_weights"], map_location=device))
        model.eval()
        logger.info(f"Model loaded from {paths['model_weights']}")
    except FileNotFoundError:
        logger.error(f"Model file not found: {paths['model_weights']}. Train the model first.")
        return
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    try:
        extractor = USLFeatureExtractor(
            paths["hand_model"],
            paths["pose_model"],
        )
    except Exception as e:
        logger.error(f"Failed to initialize extractor: {e}")
        return

    cap = cv2.VideoCapture(inf_params["camera_index"])
    if not cap.isOpened():
        logger.error(f"Failed to open camera index {inf_params['camera_index']}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, inf_params["display_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, inf_params["display_height"])

    frame_buffer = deque(maxlen=inf_params["buffer_size"])
    current_prediction = "Waiting..."
    timestamp = 0

    input_tensor = None

    logger.info("Starting live inference. Press 'q' to quit.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                break

            timestamp += 33
            landmarks = extractor.process_frame(frame, timestamp)
            frame_buffer.append(landmarks)

            if len(frame_buffer) == inf_params["buffer_size"]:
                try:
                    buffer_np = np.array([frame_buffer], dtype=np.float32)

                    if input_tensor is None or input_tensor.shape != buffer_np.shape:
                        input_tensor = torch.from_numpy(buffer_np).to(device)
                    else:
                        input_tensor.copy_(torch.from_numpy(buffer_np))

                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.softmax(output, dim=1)
                        confidence, predicted_idx = torch.max(probs, dim=1)

                        conf_val = confidence.item()
                        if conf_val > inf_params["confidence_threshold"]:
                            pred_idx = int(predicted_idx.item())
                            current_prediction = f"{actions[pred_idx]} ({conf_val:.2f})"
                        else:
                            current_prediction = "Low confidence"

                except Exception as e:
                    logger.error(f"Inference error: {e}")
                    current_prediction = "Error"

            cv2.rectangle(frame, (0, 0), (inf_params["display_width"], 40), (245, 117, 16), -1)
            cv2.putText(
                frame,
                f"USL: {current_prediction}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.imshow("USL Live Translator", frame)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        logger.info("Inference stopped")


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.get("logging.level", "INFO")),
        format=config.get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
    )
    run_inference()