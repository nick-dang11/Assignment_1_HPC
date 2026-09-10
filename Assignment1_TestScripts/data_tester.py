import numpy as np
import pandas as pd
import torch as t
from data_process1 import *
import time


if __name__ == "__main__":
    # Python approach
    print("Python Array Normalization")
    raw_data_list = load_dataset("GasProperties.csv")
    
    start_time = time.perf_counter()
    rows_py = normalize_array(raw_data_list, out_file="GasProperties_norm.csv")
    end_time = time.perf_counter()
    
    time_py = end_time - start_time
    print(f"Rows processed: {rows_py}")
    print(f"Computation time (Python lists): {time_py:.4f} seconds")

    # NumPy approach
    print("\nNumPy Array Normalization")
    raw_data_np = load_dataset_np("GasProperties.csv")
    
    start_time = time.perf_counter()
    rows_np = normalize_array_np(raw_data_np, out_file="GasProperties_norm_np.csv")
    end_time = time.perf_counter()
    
    time_np = end_time - start_time
    print(f"Rows processed: {rows_np}")
    print(f"Computation time (NumPy): {time_np:.4f} seconds")
    