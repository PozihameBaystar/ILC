import numpy as np

def flat_curve_data(
        L: float,
        a: float,
        t: np.ndarray,
) -> np.ndarray:
    data = L*(1-np.exp(-a*t))
    return data