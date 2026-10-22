import os
import time
import logging
import cv2
import numpy as np

try:
    from .config import config
except ImportError:
    from config import config
try:
    from .extractor import USLFeatureExtractor
except ImportError:
    from extractor import USLFeatureExtractor


logger = logging.getLogger(__name__)


def collect_data():
    actions = config.actions
    no_sequences = config.no_sequences
    sequence_length = config.sequence_length
    data_path = config.data_path
    paths = config.paths

    for action in actions:
        for sequence in range(no_sequences):
            os.makedirs(os.path.join(data_path, action, str(sequence)), exist_ok=True)

    try:
        extractor = USLFeatureExtractor(
            paths["hand_model"],
            paths["pose_model"],
        )
    except Exception as e:
        logger.error(f"Failed to initialize extractor: {e}")
        return

    cap = cv2.VideoCapture(config.inference_params["camera_index"])
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return

    logger.info("Starting dataset recording...")
    logger.info(f"Actions: {actions}")
    logger.info(f"Sequences per action: {no_sequences}")
    logger.info(f"Frames per sequence: {sequence_length}")

    try:
        for action_idx, action in enumerate(actions):
            for sequence in range(no_sequences):
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        logger.error("Failed to read frame from camera")
                        return

                    timestamp_ms = int(time.time() * 1000)
                    try:
                        keypoints = extractor.process_frame(frame, timestamp_ms)
                    except Exception as e:
                        logger.error(f"Error extracting features: {e}")
                        keypoints = np.zeros(225)

                    npy_path = os.path.join(data_path, action, str(sequence), f"{frame_num}.npy")
                    try:
                        np.save(npy_path, keypoints)
                    except Exception as e:
                        logger.error(f"Failed to save {npy_path}: {e}")

                    if frame_num == 0:
                        cv2.putText(
                            frame,
                            f"STARTING COLLECTION FOR: {action}",
                            (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"Sequence Number {sequence}",
                            (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                        )
                        cv2.imshow("OpenCV Feed", frame)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(
                            frame,
                            f"Collecting {action} Video #{sequence}",
                            (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                        )
                        cv2.imshow("OpenCV Feed", frame)

                    if cv2.waitKey(10) & 0xFF == ord("q"):
                        logger.info("User requested quit")
                        return

                logger.info(f"Completed {action} sequence {sequence}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        logger.info("Data collection finished")


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.get("logging.level", "INFO")),
        format=config.get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
    )
    collect_data()