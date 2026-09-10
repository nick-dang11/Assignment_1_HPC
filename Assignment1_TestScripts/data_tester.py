import numpy as np
import pandas as pd
import torch as t
from data_process1 import *
import time


if __name__ == "__main__":

    GasProperties = load_dataset("GasProperties.csv")

    print("TEST load_dataset")
    print("Number of rows:", len(GasProperties))
    print("First row:", GasProperties[0])
    print("Type of dataset:", type(GasProperties))
    print("Type of first value:", type(GasProperties[0][0]))