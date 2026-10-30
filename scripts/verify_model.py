"""Load models/best_model.joblib in a fresh Python process and check inference."""

import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import load_splits, split_xy  # noqa: E402

model = joblib.load("models/best_model.joblib")
_train, val, _test = load_splits()
val_x, _ = split_xy(val)
proba = model.predict_proba(val_x.iloc[:200])[:, 1]
reference = np.load("models/reference_proba.npy")

assert proba.shape == (200,), proba.shape
assert np.allclose(proba, reference, atol=1e-9), "predictions differ from notebook"
print("OK: model loads in a fresh process and predictions match the notebook")
