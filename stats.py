"""
This module provides the statistical tools for the computation of indicators correlation. It is
mainly used to realize principal component analysis. This method performs this analysis along with
data manipulation to compute confidence intervals.
"""

from numpy import delete, array
from numpy.typing import NDArray
from numpy.random import default_rng
from typing import List

def generate_bootstraped_dataset(data: NDArray) -> NDArray:
    """
    Generates one bootstrap sampled from the data provided in the parameters. The generated
    bootstrap sample is drawn from the dataset.

    Args:
        - data: The empiric dataset to which bootstraped observations will be drawn from.

    Returns: A bootstraped dataset to which each observation from the bootstrap dataset can be found
        in the empiric dataset.
    """

    rng = default_rng()
    bootstraped = []
    for _ in data:
        row_index = rng.integers(0, len(data))
        bootstraped.append(data[row_index])

    return array(bootstraped)

def jacknife(data: NDArray) -> List[NDArray]:
    """
    Applies the jacknife algorithm to the dataset.

    This algorithm consist in creating samples in which one observation is removed.

    Args:
        - data: The data to jacknife.

    Returns: A list of jacknifed sample, the `i`-th entry of the returned array is a jacknife sample
        to which the `i`-th row was removed from the empirical dataset.
    """
    return [delete(data, i, 0) for i in range(len(data))]
