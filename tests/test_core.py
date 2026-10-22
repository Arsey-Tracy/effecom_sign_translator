import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock

from src.config import Config
from src.train import USLClassifier


class TestConfig:
    def test_config_singleton(self):
        c1 = Config()
        c2 = Config()
        assert c1 is c2

    def test_config_loads_actions(self):
        c = Config()
        actions = c.actions
        assert isinstance(actions, list)
        assert len(actions) == 3
        assert "oli_otya" in actions

    def test_config_model_params(self):
        c = Config()
        params = c.model_params
        assert params["input_dim"] == 225
        assert params["hidden_dim"] == 64
        assert params["num_layers"] == 2
        assert params["dropout"] == 0.2

    def test_config_training_params(self):
        c = Config()
        params = c.training_params
        assert params["epochs"] == 100
        assert params["batch_size"] == 16
        assert params["learning_rate"] == 0.001
        assert params["device"] == "cpu"

    def test_config_inference_params(self):
        c = Config()
        params = c.inference_params
        assert params["confidence_threshold"] == 0.75
        assert params["buffer_size"] == 30
        assert params["camera_index"] == 0

    def test_config_paths(self):
        c = Config()
        paths = c.paths
        assert "hand_model" in paths
        assert "pose_model" in paths
        assert "model_weights" in paths


class TestUSLClassifier:
    def test_model_initialization(self):
        model = USLClassifier(
            input_dim=225,
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
            num_classes=3,
        )
        assert isinstance(model, USLClassifier)
        assert model.lstm.input_size == 225
        assert model.lstm.hidden_size == 64
        assert model.lstm.num_layers == 2
        assert model.fc.out_features == 3

    def test_model_forward_pass(self):
        model = USLClassifier(num_classes=3)
        model.eval()

        batch_size = 2
        seq_len = 30
        input_dim = 225

        x = torch.randn(batch_size, seq_len, input_dim)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 3)

    def test_model_output_values(self):
        model = USLClassifier(num_classes=3)
        model.eval()

        x = torch.zeros(1, 30, 225)
        with torch.no_grad():
            output = model(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_model_different_num_classes(self):
        for num_classes in [2, 3, 5, 10]:
            model = USLClassifier(num_classes=num_classes)
            assert model.fc.out_features == num_classes

            x = torch.randn(1, 10, 225)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (1, num_classes)


class TestFeatureExtractorOutput:
    def test_feature_vector_dimension(self):
        expected_dim = 33 * 3 + 2 * 21 * 3
        assert expected_dim == 225

    def test_empty_landmarks_handling(self):
        pose_arr = np.zeros(33 * 3)
        left_hand = np.zeros(21 * 3)
        right_hand = np.zeros(21 * 3)
        combined = np.concatenate([pose_arr, left_hand, right_hand])
        assert combined.shape == (225,)


class TestDataLoading:
    def test_label_map_creation(self):
        actions = ["oli_otya", "oli_bulungi", "webale"]
        label_map = {label: num for num, label in enumerate(actions)}
        assert label_map["oli_otya"] == 0
        assert label_map["oli_bulungi"] == 1
        assert label_map["webale"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])