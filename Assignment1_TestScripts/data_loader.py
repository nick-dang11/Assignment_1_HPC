'''
This file is where you migrate all the functions that you made in data_process1 and data_process2. 
'''
import numpy as np
import pandas as pd
import csv
import time # Import the time module
from typeguard import typechecked

@typechecked
def load_dataset(filename: str) -> list[list[float]]:
    """
    Loads a dataset from a CSV file into a list of lists of floats.
    The first row (header) is skipped.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        list[list[float]]: A list of lists, where each inner list represents a row
                           of the dataset with float values.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.

    Note: If you are stuck on this, I recommend looking through the python csv library.
    """
    GasProperties = []
    with open(filename, "r") as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            converted_row = [float(value) for value in row]
            GasProperties.append(converted_row)
    return GasProperties


@typechecked
def load_dataset_np(filename: str) -> np.ndarray:
    """
    Loads a dataset from a CSV file into a NumPy array.
    The first row (header) is skipped.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        np.ndarray: A NumPy array representing the dataset.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    
    Note: Numpy has a very useful csv file reader called genfromtxt.
    """
    GasProperties = np.genfromtxt(
        filename,
        delimiter=",",
        skip_header=1
    )
    return GasProperties
    #raise NotImplementedError()

@typechecked
def normalize_array(arr: list[list[float]], out_file: str | None = None) -> int:
    """
    Normalizes the input array (list of lists) using min-max normalization
    and filters out outliers based on standard deviation.
    The last column is assumed to be the target variable and is not normalized.
    Optionally writes the normalized data to a new CSV file.

    Args:
        arr (list[list[float]]): The input dataset as a list of lists of floats.
        out_file (str, optional): The path to the output CSV file. Defaults to None.

    Returns:
        int: The number of rows processed after normalization and filtering.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    
    Note: Probably the most complicated function to write because you cant use numpy.
    I would spend some time on this to make sure that all the equations for metrics are correct.
    """
     #save the metrics for each column in a list
    means = []
    minimums = []
    maximums = []
    standard_deviations = []

    for column in range(4):
        values = [row[column] for row in arr]
        mean = sum(values)/len(values)
        minimum = min(values)
        maximum = max(values)

        squared_differences_sum = sum((value - mean) ** 2 for value in values)
        variance = squared_differences_sum / len(values)
        standard_deviation = variance ** 0.5
        means.append(mean)
        minimums.append(minimum)
        maximums.append(maximum)
        standard_deviations.append(standard_deviation)

    #outliers to be filtered out
    filtered_rows = []

    for row in arr:
        is_outlier = False
        for column in range(4):
            if abs(row[column] - means[column]) > 2 * standard_deviations[column]:
                is_outlier = True
                break
            
        if not is_outlier:
            filtered_rows.append(row)

    normalized_rows = []

    for row in filtered_rows:
        new_row = []

        for column in range(4):
            denominator = maximums[column] - minimums[column]

            if denominator == 0:
                normalized_value = 0.0  # or any other value you want to assign in this case
            else:
                normalized_value = (row[column] - means[column]) / denominator

            new_row.append(normalized_value)
# Append the target variable without normalization

        new_row.append(row[4])

        normalized_rows.append(new_row)

    if out_file is not None:
        with open(out_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["T", "P", "TC", "SV", "idx"])
            writer.writerows(normalized_rows)

    return len(normalized_rows)
    
    #raise NotImplementedError()

@typechecked
def normalize_array_np(arr: np.ndarray, out_file: str | None = None) -> int:
    """
    Normalizes the input NumPy array using min-max normalization
    and filters out outliers based on standard deviation.
    The last column is assumed to be the target variable and is not normalized.
    Optionally writes the normalized data to a new CSV file.

    Args:
        arr (np.ndarray): The input dataset as a NumPy array.
        out_file (str, optional): The path to the output CSV file. Defaults to None.

    Returns:
        int: The number of rows processed after normalization and filtering.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.

    Note: The same function as normalize_array but using numpy to calculate the metrics.
    This function should be almost copy and paste with numpy functions.
    """
    features = arr[:, :4] # all rows and first 4 columns 

    means = np.mean(features, axis=0)
    minimums = np.min(features, axis=0)
    maximums = np.max(features, axis=0)
    standard_deviations = np.std(features, axis=0)

    outliers = np.abs(features - means) > 2 * standard_deviations
    row_has_outlier = np.any(outliers, axis=1)
    filtered_arr = arr[~row_has_outlier]

    denominator = maximums - minimums
    denominator[denominator == 0] = 1  # Avoid division by zero

    normalized_features = (filtered_arr[:, :4] - means) / denominator

    idx_column = filtered_arr[:, 4:].reshape(-1, 1)
    normalized_array = np.hstack((normalized_features, idx_column))

    if out_file is not None:
        np.savetxt(out_file, normalized_array, delimiter=",", header="T,P,TC,SV,idx", comments='')

    return len(normalized_array)
    #raise NotImplementedError()

@typechecked
def load_dataset_pd(filename: str) -> pd.DataFrame:
    """
    Loads normalized data from a CSV file into a Pandas DataFrame.

    Args:
        filename (str): The path to the CSV file containing normalized data.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the loaded data.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.

    Note: This is a 1-line solution, if you spend more than 10 mins on this, you may be overthinking.
    """
    return pd.read_csv(filename)

@typechecked
def split_xy(df: pd.DataFrame, y_axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits a Pandas DataFrame into feature (X) and target (Y) NumPy arrays.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame.
        y_axis (int, optional): The column index of the target variable (Y).
                                Defaults to -1 (last column).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
                                       X (features) and Y (target).

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    Note: A 2-line solution but can be tricky if you are not familiar with pandas.
    """
    x_data = df.drop(df.columns[y_axis], axis = 1).to_numpy()
    y_data = df.iloc[:, y_axis].to_numpy()
    return x_data, y_data

@typechecked
def split_training_test(
        X_data: np.ndarray, 
        Y_data: np.ndarray, 
        split: float = 0.8
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the input feature (X) and target (Y) NumPy arrays into training and testing sets.

    Args:
        X_data (np.ndarray): The input feature array.
        Y_data (np.ndarray): The input target array.
        split (float, optional): The proportion of data to be used for training.
                                 Defaults to 0.8 (80% training, 20% testing).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing four NumPy arrays:
                                                                X_train, Y_train, X_test, Y_test.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    Note: If you are stuck, look into list slicing. 
    """
    split_idx = int(len(X_data) * split)
    return X_data[:split_idx], Y_data[:split_idx], X_data[split_idx:], Y_data[split_idx:]