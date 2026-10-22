import logging
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from .config import config
except ImportError:
    from config import config


logger = logging.getLogger(__name__)


class USLFeatureExtractor:
    def __init__(
        self,
        hand_model_path: str = None,
        pose_model_path: str = None,
    ):
        paths = config.paths
        hand_model_path = hand_model_path or paths["hand_model"]
        pose_model_path = pose_model_path or paths["pose_model"]

        try:
            hand_options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=hand_model_path),
                num_hands=2,
                running_mode=vision.RunningMode.VIDEO,
            )
            self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)
            logger.info(f"Hand landmarker loaded from {hand_model_path}")
        except Exception as e:
            logger.error(f"Failed to load hand landmarker: {e}")
            raise

        try:
            pose_options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=pose_model_path),
                running_mode=vision.RunningMode.VIDEO,
            )
            self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
            logger.info(f"Pose landmarker loaded from {pose_model_path}")
        except Exception as e:
            logger.error(f"Failed to load pose landmarker: {e}")
            raise

    def process_frame(self, frame: np.ndarray, timestamp_ms: int) -> np.ndarray:
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            hand_res = self.hand_detector.detect_for_video(mp_image, timestamp_ms)
            pose_res = self.pose_detector.detect_for_video(mp_image, timestamp_ms)

            pose_arr = np.zeros(33 * 3)
            if pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0:
                pose_arr = np.array(
                    [[lm.x, lm.y, lm.z] for lm in pose_res.pose_landmarks[0]]
                ).flatten()

            left_hand = np.zeros(21 * 3)
            right_hand = np.zeros(21 * 3)

            if hand_res.hand_landmarks and hand_res.handedness:
                for idx, handness in enumerate(hand_res.handedness):
                    label = handness[0].category_name
                    landmarks = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_res.hand_landmarks[idx]]
                    ).flatten()
                    if label == "Left":
                        left_hand = landmarks
                    else:
                        right_hand = landmarks

            return np.concatenate([pose_arr, left_hand, right_hand])

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return np.zeros(225)

    def close(self):
        try:
            if hasattr(self, "hand_detector") and self.hand_detector:
                self.hand_detector.close()
            if hasattr(self, "pose_detector") and self.pose_detector:
                self.pose_detector.close()
            logger.info("Detectors closed")
        except Exception as e:
            logger.error(f"Error closing detectors: {e}")