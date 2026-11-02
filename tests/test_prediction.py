"""ML feature construction."""

import numpy as np
import pytest

from app.services.prediction_service import pair_features


def test_pair_features_shape_and_cosine():
    vec = np.ones(4)
    features = pair_features(vec, vec, 9)
    assert features.shape == (6,)
    assert features[4] == pytest.approx(1.0)
    assert features[5] == pytest.approx(np.log1p(9))
    assert pair_features(np.zeros(4), vec, 0)[4] == 0.0
