import numpy as np

def flat_curve_data1(
        L: float,
        a: float,
        t: np.ndarray,
) -> np.ndarray:
    data = L * (1-np.exp(-a*t))**2
    return data

def flat_curve_data2(
        L: float,
        a: float,
        t: np.ndarray,
) -> np.ndarray:
    data = L * np.tanh(a*t)**2
    return data