from typing import Sequence
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

def compute_error(rho_hat: np.ndarray, rho_true: np.ndarray, err_type: str = "fro_sq") -> float:
    """
    Computer error between rho hat and true rho
    """
    if err_type == "fro_sq":
        return np.linalg.norm(rho_hat - rho_true)**2 #type: ignore
    elif err_type == "fro":
        return np.linalg.norm(rho_hat - rho_true) #type: ignore
    elif err_type == "MSE":
        d0, d1 = rho_hat.shape
        return np.trace((rho_hat - rho_true) @ np.conj((rho_hat - rho_true).T)) / (d0*d1)
    else:
        raise ValueError("No such error type")
    
def dump_run_information(path: str, d: dict[str, Sequence|np.ndarray]):
    """
    Write information from experiment to file given a dict `d` to `path`.csv 
    """
    df = pd.DataFrame.from_dict(d)
    df.to_csv(path + ".csv")

def dump_run_information_from_tensors(tensor_prob: np.ndarray, tensor_pl: np.ndarray, cols: dict[str, ArrayLike], path: str, map_colidx_colname: dict[int, str]|None=None):
    """
    tensor_prob: np.ndarray with accuracy for prob
    tensor_pl: np.ndarray with accuracy for pl
    cols:  dict[str, list[float]] which maps the name of each dimension to its range of values
    We then combine them to create an output in the format:
    col1 | col2 | col3 | acc_prob | acc_pl
    1      1     1       0.2        0.3
    1      1     2       0.1        0.4
    ...
    """
    if map_colidx_colname is None:
        map_colidx_colname = {i: col for i, col in enumerate(cols.keys())}
    df_list = []
    for ((idx_prob, val_prob), (idx_pl, val_pl)) in zip(np.ndenumerate(tensor_prob), np.ndenumerate(tensor_pl)):
            cols_values_except_acc = []
            for k, idx_in_col in enumerate(idx_prob):
                col = map_colidx_colname[k]
                col_value = cols[col][idx_in_col] # type: ignore
                cols_values_except_acc.append(col_value)
            df_list.append(cols_values_except_acc + [val_prob, val_pl])
    df = pd.DataFrame(df_list, columns=list(cols.keys()) + ["acc_prob", "acc_pl"])
    df.to_csv(path + ".csv")


def dump_run_information_from_tensors3(tensor_prob: np.ndarray, tensor_pl: np.ndarray, tensor3: np.ndarray, algo3_name: str, cols: dict[str, ArrayLike], path: str, map_colidx_colname: dict[int, str]|None=None):
    """
    tensor_prob: np.ndarray with accuracy for prob
    tensor_pl: np.ndarray with accuracy for pl
    cols:  dict[str, list[float]] which maps the name of each dimension to its range of values
    We then combine them to create an output in the format:
    col1 | col2 | col3 | acc_prob | acc_pl
    1      1     1       0.2        0.3
    1      1     2       0.1        0.4
    ...
    """
    if map_colidx_colname is None:
        map_colidx_colname = {i: col for i, col in enumerate(cols.keys())}
    df_list = []
    for ((idx_prob, val_prob), (idx_pl, val_pl), (idx3, val3)) in zip(np.ndenumerate(tensor_prob), np.ndenumerate(tensor_pl), np.ndenumerate(tensor3)):
            cols_values_except_acc = []
            for k, idx_in_col in enumerate(idx_prob):
                col: str = map_colidx_colname[k]
                col_value = cols[col][idx_in_col] # type: ignore
                cols_values_except_acc.append(col_value)
            df_list.append(cols_values_except_acc + [val_prob, val_pl, val3])
    df = pd.DataFrame(df_list, columns=list(cols.keys()) + ["acc_prob", "acc_pl", "acc_" + algo3_name])
    df.to_csv(path + ".csv")

def dump_run_information_from_tensors4(tensor_prob: np.ndarray, tensor_pl: np.ndarray, tensor3: np.ndarray, tensor4: np.ndarray, algo3_name: str, algo4_name: str, cols: dict[str, Sequence|np.ndarray], path: str, map_colidx_colname: dict[int, str]|None=None):
    """
    tensor_prob: np.ndarray with accuracy for prob
    tensor_pl: np.ndarray with accuracy for pl
    cols:  dict[str, list[float]] which maps the name of each dimension to its range of values
    We then combine them to create an output in the format:
    col1 | col2 | col3 | acc_prob | acc_pl
    1      1     1       0.2        0.3
    1      1     2       0.1        0.4
    ...
    """
    if map_colidx_colname is None:
        map_colidx_colname = {i: col for i, col in enumerate(cols.keys())}
    df_list = []
    for ((idx_prob, val_prob), (idx_pl, val_pl), (idx3, val3), (idx4, val4)) in zip(np.ndenumerate(tensor_prob), np.ndenumerate(tensor_pl), np.ndenumerate(tensor3), np.ndenumerate(tensor4)):
            cols_values_except_acc = []
            for k, idx_in_col in enumerate(idx_prob):
                col = map_colidx_colname[k]
                col_value = cols[col][idx_in_col]
                cols_values_except_acc.append(col_value)
            df_list.append(cols_values_except_acc + [val_prob, val_pl, val3, val4])
    df = pd.DataFrame(df_list, columns=list(cols.keys()) + ["acc_prob", "acc_pl", "acc_" + algo3_name, "acc_" + algo4_name])
    df.to_csv(path + ".csv")



def dump_run_information_from_tensor(algo_name: str, tensor: np.ndarray, cols: dict[str, Sequence|np.ndarray], path: str, avgs: list, map_colidx_colname: dict[int, str]|None=None):
    """
    tensor_prob: np.ndarray with accuracy for prob
    tensor_pl: np.ndarray with accuracy for pl
    cols:  dict[str, list[float]] which maps the name of each dimension to its range of values
    We then combine them to create an output in the format:
    col1 | col2 | col3 | acc_prob | acc_pl
    1      1     1       0.2        0.3
    1      1     2       0.1        0.4
    ...
    """
    if map_colidx_colname is None:
        map_colidx_colname = {i: col for i, col in enumerate(cols.keys())}
    df_list = []
    idx_avgs = 0
    for (idx_prob, val_acc) in np.ndenumerate(tensor):
            cols_values_except_acc = []
            for k, idx_in_col in enumerate(idx_prob):
                col = map_colidx_colname[k]
                col_value = cols[col][idx_in_col]
                cols_values_except_acc.append(col_value)
            df_list.append(cols_values_except_acc + [avgs[idx_avgs], val_acc])
            idx_avgs += 1
    df = pd.DataFrame(df_list, columns=list(cols.keys()) + ["sample_avg", "acc_" + algo_name])
    df.to_csv(path + ".csv")