import numpy as np
from typing import Optional, Union


def l2_err_norm(
    true: np.ndarray,
    pred: np.ndarray,
    axis: Optional[Union[int, tuple]] = None,
) -> np.ndarray:
    """Relative L2 error: ||true - pred|| / ||true||."""
    return np.linalg.norm(true - pred, axis=axis) / np.linalg.norm(true, axis=axis)
