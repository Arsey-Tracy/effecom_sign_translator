import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import numpy as np


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture
def sample_actions():
    return ["oli_otya", "oli_bulungi", "webale"]


@pytest.fixture
def sample_sequences():
    return 5


@pytest.fixture
def sample_sequence_length():
    return 30


@pytest.fixture
def dummy_features():
    return np.random.randn(30, 225).astype(np.float32)