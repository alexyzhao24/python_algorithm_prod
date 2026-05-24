import test
import random
import sys

## without install the package locally: pip install -e .
# you then need to do the add project root to path
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add project root to path
from python_algo_production.algo.lralgo import *

def test_lr_hist_naive_basic():
    assert lr_hist_naive(np.array([2, 1, 5, 6, 2, 3]))[0] == 10
    assert lr_hist_naive(np.array([1, 3, 7, 4, 0, 10, 5, 2]))[0] == 10
    assert lr_hist_naive(np.array([1, 4, 7, 4, 0, 10, 5, 2]))[0] == 12
    assert lr_hist_naive(np.array([24, 17, 20, 12, 5]))[0] == 51


def test_lr_hist_naive_random():
    for iter in range(5):
        # Generate a random array of integers from 0 to 100 (inclusive)
        random_hist = np.random.randint(0, 100, size=10)   ## range 0 to 100
        lr_hist_naive(random_hist)


def test_lr_hist_basic():

    print("sys.path during debug:", sys.path)
    assert lr_hist(np.array([2, 1, 5, 6, 2, 3])) == 10
    assert lr_hist(np.array([1, 3, 7, 4, 0, 10, 5, 2])) == 10
    assert lr_hist(np.array([1, 4, 7, 4, 0, 10, 5, 2])) == 12
    assert lr_hist(np.array([24, 17, 20, 12, 5])) == 51


def test_lr_hist_random():
    for iter in range(5):
        # Generate a random array of integers from 0 to 100 (inclusive)
        random_hist = np.random.randint(0, 100, size=10)   ## range 0 to 100
        assert lr_hist(random_hist) == lr_hist_naive(random_hist)[0]
