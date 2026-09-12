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
    import sys
    vp = sys.modules[__name__]
    df_norm = pd.read_csv("GasProperties_norm.csv")
    X_data = df_norm[['T', 'P', 'TC', 'SV']].to_numpy()

    print("Starting benchmarks...")

    # Create a 4-element query vector matching X_data's 4 feature columns
    Y_vector = np.array([1.0, 0.5, -0.5, 1.0])

    # 2(a): Python Loop Dot Product Benchmark
    start = time.perf_counter()
    best_py = vp.find_largest_dot_product_py(X_data.tolist(), Y_vector.tolist())
    time_py = time.perf_counter() - start
    print(f"Python Loop -> Index: {best_py} | Time: {time_py:.4f} s")

    # 2(b): NumPy Dot Product Benchmark (Vectorized)
    start = time.perf_counter()
    best_np = vp.find_largest_dot_product_np(X_data, Y_vector)
    time_np = time.perf_counter() - start

    np_dot_values = X_data @ Y_vector
    max_val_np = np_dot_values[best_np]
    print(f"NumPy Vectorized -> Index: {best_np} | Max Dot Product: {max_val_np:.4f} | Time: {time_np:.4f} s")

    # 2(c): Matrix Multiplication Precision (32-bit vs 64-bit)
    X_32 = X_data.astype(np.float32)
    X_64 = X_data.astype(np.float64)

    start = time.perf_counter()
    res_32 = vp.mat_mul_np(X_32.T, X_32)
    print(f"NumPy 32-bit MatMul Time: {time.perf_counter() - start:.4f} s")

    start = time.perf_counter()
    res_64 = vp.mat_mul_np(X_64.T, X_64)
    print(f"NumPy 64-bit MatMul Time: {time.perf_counter() - start:.4f} s")

    # 2(d): PyTorch GPU Acceleration (64-bit)
    A_gpu = t.tensor(X_64, dtype=t.float64, device='cuda')
    t.cuda.synchronize()
    start = time.perf_counter()
    res_gpu = vp.mat_mul_t(A_gpu.T, A_gpu)
    t.cuda.synchronize()
    print(f"PyTorch GPU 64-bit MatMul Time: {time.perf_counter() - start:.4f} s")


