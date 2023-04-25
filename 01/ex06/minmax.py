import numpy as np
from numpy import ndarray


def minmax(x: ndarray) -> ndarray:
    """
    Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
    Args:
        x: has to be an numpy.ndarray, a vector.
    Returns:
        x' as a numpy.ndarray.
        None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
        This function shouldn't raise any Exception.
    """
    return (x - x.min())/(x.max() - x.min())
