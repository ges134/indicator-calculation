"""
This module provides the statistical tools for computing indicator correlations. It is mainly used
to realize principal component analysis. This method performs the analysis and data manipulation to
compute confidence intervals.
"""

from numpy import delete, array, mean, std, corrcoef, argmax, sign, newaxis, copy
from numpy.linalg import eig
from numpy.typing import NDArray
from numpy.random import default_rng
from typing import List, Tuple
from scipy.stats import anderson, boxcox

def generate_bootstraped_dataset(data: NDArray) -> NDArray:
    """
    Generates a single bootstrap sample from the data provided in the parameters. The generated
    bootstrap sample is drawn from the dataset.

    Args:
        - data: The empirical dataset from which bootstraped observations will be drawn.

    Returns: A bootstrapped dataset, where each observation from the bootstrap dataset can be found
        in the empirical dataset.
    """

    rng = default_rng()
    bootstraped = []
    for _ in data:
        row_index = rng.integers(0, len(data))
        bootstraped.append(data[row_index])

    return array(bootstraped)

def jacknife(data: NDArray) -> List[NDArray]:
    """
    Applies the jackknife algorithm to the dataset.

 This algorithm consists of creating samples by removing one observation.

    Args:
        - data: The data to jackknife.

    Returns: A list of jackknifed samples; the `i`-th entry of the returned array is the jackknifed
        sample obtained by removing the `i`-th row from the empirical dataset.
    """
    return [delete(data, i, 0) for i in range(len(data))]

def apply_pca(data: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Does the PCA on the data given as a parameter.

    This method, more precisely, performs a PCA on a correlation matrix without any transformation.

    Most of the code is taken from the following blog post along with variable changes:
    https://bagheri365.github.io/blog/Principal-Component-Analysis-from-Scratch/

    Args:
        - data: The dataset to compute the PCA. The data should be complete and have no empty
            entries.

    Returns: The results of the PCA, in the form of a tuple. The first value of the tuple gives the
        eigenvalues. The second gives the eigenvectors, and the last gives the explained variance by
        each principal component.
    """
    copied_data = array(data)

    average = mean(copied_data, axis=0)
    standard_deviation = std(copied_data, axis=0)
    copied_data = (copied_data - average) / standard_deviation

    correlation_matrix = corrcoef(copied_data.T)
    eigen_values, eigen_vectors = eig(correlation_matrix)

    # Adjusting the eigenvectors (loadings) that are largest in absolute value to be positive
    max_abs_index = argmax(abs(eigen_vectors), axis=0)
    signs = sign(eigen_vectors[max_abs_index, range(eigen_vectors.shape[0])])
    eigen_vectors = eigen_vectors*signs[newaxis,:]
    eigen_vectors = eigen_vectors.T

    # We first make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = [(abs(eigen_values[i]), eigen_vectors[i,:]) for i in range(len(eigen_values))]

    # Then, we sort the tuples from the highest to the lowest based on eigenvalues magnitude
    eigen_pairs.sort(key=lambda x: x[0], reverse=True)

    # For further usage
    eigen_values_sorted = array([x[0] for x in eigen_pairs])
    eigen_vectors_sorted = array([x[1] for x in eigen_pairs])

    eigen_values_total = sum(eigen_values)
    explained_variance = [(i / eigen_values_total) for i in eigen_values_sorted]

    return eigen_values_sorted, eigen_vectors_sorted.T, explained_variance

def correlation_matrix_between_pcas(eigen_vectors_a: NDArray, eigen_vectors_b: NDArray) -> NDArray:
    """
    Computes a correlation matrix between the eigenvectors of two different PCAs.

    Args:
        - eigen_vectors_a: The eigen vectors of the first PCA.
        - eigen_vectors_b: The eigen vectors of the second PCA.

    Returns: A matrix in which the elements correspond to the correlation of the eigen vectors. The
        rows represent the eigen vectors of the principal component A (`eigen_vectors_a` parameter)
        while the columns represent the eigen vectors of the principal component B 
        (`eigen_vectors_b` parameter). Hence, the element `[0][1]` represents the correlation
        between the eigenvector of the first principal component from the A set and the eigenvector
        from the second component from the B set.
    """

    _, nb_cols = eigen_vectors_a.shape
    correlation_matrix = []
    for i in range(nb_cols):
        row = []
        for j in range(nb_cols):
            correlation = corrcoef(eigen_vectors_a[:, i], eigen_vectors_b[:, j])[0][1]
            row.append(correlation)

        correlation_matrix.append(row)

    return array(correlation_matrix)

def test_for_normality(data: NDArray) -> List[bool]:
    """
    Looks if each column of the data follows a normal distribution following the Anderson-Darling
    test.

    Args:
        - data: The dataset to test for normality. Each row should represent observations, while
            each column should represent parameters. In this program, each row represents countries,
            and each column represents indicators.
    
    Returns: An array in which each element corresponds to the normality of the associated column.
        If it follows a normal distribution, the element will be `True`.
    """
    is_normal = []
    _, columns = data.shape
    for i in range(columns):
        normality_test_results = anderson(data[:, i])
        is_normal.append(
            normality_test_results.statistic < normality_test_results.critical_values[2]
        )

    return is_normal

def boxcox_transform(data: NDArray) -> NDArray:
    """
    Transform the rows in the dataset that are not normally distributed. The method performs the
    normality test.

    Args:
        - data: The data to transform.
    
    Returns: The transformed data where columns are replaced with their normalized values if they do
        not follow a normal distribution and are unchanged if they do. If the data have not been
        modified by the method, it means that all variables are normally distributed.
    """

    copied_data = copy(data)
    is_normal = test_for_normality(copied_data)
    for i, normality in enumerate(is_normal):
        if not normality:
            column = copied_data[:, i]
            if 0 not in column:
                transformed, _ = boxcox(column)
                copied_data[:, i] = transformed

    return copied_data
