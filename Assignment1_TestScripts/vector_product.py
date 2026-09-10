import numpy as np
import torch as t
from typeguard import typechecked
import pandas as pd
import time 

@typechecked
def dot_product(a: list[float], b: list[float]) -> float:
    """
    Calculates the dot product of two lists of floats.

    Args:
        a (list[float]): The first list of float values.
        b (list[float]): The second list of float values.

    Returns:
        float: The dot product of the two lists.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    
    Note: You can use standard for-loops or list comprehension if you want to be fancy.
    If you want a challenge, try writing this in 1 line.
    """
    return sum(x * y for x, y in zip(a,b))
    #raise NotImplementedError()

@typechecked
def find_largest_dot_product_py(X_data: list[list[float]], Y_data: list[float]) -> int:
    """
    Finds the index of the row in X_data that has the largest dot product with Y_data.
    This is the pure Python implementation.

    Args:
        X_data (list[list[float]]): A list of lists representing the input features.
        Y_data (list[float]): A list representing the target values.

    Returns:
        int: The index of the row in X_data with the largest dot product.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.

    Note: I recommend using the dot_product function you wrote above to complete this.
    The implementation for this function is pretty straightforward.
    """
    largest_dot = float('-inf')
    best_index = -1
    for i, vector in enumerate(X_data):
        current_dot = dot_product(vector, Y_data)
        if current_dot > largest_dot:
            largest_dot = current_dot
            best_index = i
    return best_index
    #raise NotImplementedError()

@typechecked
def dot_product_np(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculates the dot product of two NumPy arrays.

    Args:
        a (np.ndarray): The first NumPy array.
        b (np.ndarray): The second NumPy array.

    Returns:
        float: The dot product of the two arrays.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.

    Note: This is a 1-line solution, if you spend more than 10 mins on this, you may be overthinking.
    """
    return float(np.dot(a, b))
    #raise NotImplementedError()

@typechecked
def find_largest_dot_product_np(X_data: np.ndarray, Y_data: np.ndarray) -> int:
    """
    Finds the index of the row in X_data that has the largest dot product with Y_data.
    This is the NumPy implementation.

    Args:
        X_data (np.ndarray): A NumPy array representing the input features.
        Y_data (np.ndarray): A NumPy array representing the target values.

    Returns:
        int: The index of the row in X_data with the largest dot product.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    
    Note: This might be a little more tricky to do, the solution itself is pretty short (3 lines) 
    but finding the right function in numpy might be difficult.
    """
    dot_products = X_data @ Y_data
    largest_index = np.argmax(dot_products)
    return int(largest_index)
    #raise NotImplementedError()

@typechecked
def mat_mul_np(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Performs matrix multiplication on two NumPy arrays.

    Args:
        A (np.ndarray): The first NumPy array (matrix).
        B (np.ndarray): The second NumPy array (matrix).

    Returns:
        np.ndarray: The result of the matrix multiplication.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    
    Note: This is a 1-line solution, if you spend more than 10 mins on this, you may be overthinking.
    """
    return A @ B
    

@typechecked
def mat_mul_t(A: t.Tensor, B: t.Tensor) -> t.Tensor:
    """
    Performs matrix multiplication on two PyTorch tensors.

    Args:
        A (t.Tensor): The first PyTorch tensor (matrix).
        B (t.Tensor): The second PyTorch tensor (matrix).

    Returns:
        t.Tensor: The result of the matrix multiplication.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.

    Note: This is a 1-line solution, if you spend more than 10 mins on this, you may be overthinking.
    """
    return A @ B

@typechecked
def dot_product_t(a: t.Tensor, b: t.Tensor) -> t.tensor:
    """
    Calculates the dot product of two PyTorch tensors.

    Args:
        a (t.Tensor): The first PyTorch tensor.
        b (t.Tensor): The second PyTorch tensor.

    Returns:
        t.Tensor: The dot product of the two tensors.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    
    Note: This is a 1-line solution, if you spend more than 10 mins on this, you may be overthinking.
    """
    return t.dot(a,b)



if __name__ == "__main__":

    df_norm = pd.read_csv("GasProperties_norm.csv")

    X_data_np = df_norm[['T', 'P', 'TC', 'SV']].to_numpy().T 
    Y_data_np = df_norm['idx'].to_numpy()

    X_data_list = [
        df_norm['T'].tolist(), 
        df_norm['P'].tolist(), 
        df_norm['TC'].tolist(), 
        df_norm['SV'].tolist()
    ]
    Y_data_list = df_norm['idx'].tolist()
    
    start_time = time.perf_counter()
    best_py_idx = find_largest_dot_product_py(X_data_list, Y_data_list)
    end_time = time.perf_counter()
    print(f"Loop-based dot product time: {end_time - start_time:.6f} seconds")
    print(f"Best column index (Python loop): {best_py_idx}")
    print("Matrix Multiplication Benchmarks")

    start_time = time.perf_counter()
    best_np_idx = find_largest_dot_product_np(X_data_np, Y_data_np)
    end_time = time.perf_counter()
    print(f"NumPy dot product time: {end_time - start_time:.6f} seconds")
    print(f"Best column index (NumPy): {best_np_idx}")

    
    X_np = df_norm[['T', 'P', 'TC', 'SV']].to_numpy()
    
    X_32 = X_np.astype(np.float32)
    X_64 = X_np.astype(np.float64)

    start_time = time.perf_counter()
    res = mat_mul_np(X_32.T, X_32)
    end_time = time.perf_counter()
    
    print(f"Matrix multiplication completed successfully in {end_time - start_time:.6f} seconds.")
    print("Result shape:", res.shape)
