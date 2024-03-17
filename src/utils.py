import time

from typing import Callable, Iterable
import numpy as np
import pandas as pd

def time_run(f: Callable):
    """Times the execeution of a function
    Args:
        f (Callable): some function to time
    Returns:
        Tuple[float, Any] time and return values from function f 
    """
    tic = time.perf_counter()
    r_val = f()
    tac = time.perf_counter()
    return tac - tic, r_val 

def compute_error(rho_hat: np.ndarray, rho_true: np.ndarray, err_type: str = "fro_sq"):
    """
    """
    if err_type == "fro_sq":
        return np.linalg.norm(rho_hat - rho_true)**2
    elif err_type == "fro":
        return np.linalg.norm(rho_hat - rho_true)
    elif err_type == "MSE":
        d0, d1 = rho_hat.shape
        return np.trace((rho_hat - rho_true) @ np.conj((rho_hat - rho_true).T)) / (d0*d1)
    else:
        raise ValueError("No such error type")
    
def dump_run_information(path: str, d: dict[str, Iterable]):
    df = pd.DataFrame.from_dict(d)
    df.to_csv(path + ".csv")