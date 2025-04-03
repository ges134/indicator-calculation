"""
This module computes the degree of independances from the indicators. The degrees follow the
interpretation of principal component analysis. The closer an indicator is to 90 degrees in regards
to the eigen vector of the first two principal components, the more independant it is. This module
allows also the computation of the PCA.
"""

from numpy import array, corrcoef, abs, argmax, sign, newaxis, rad2deg, acos, dot, mean, std
from numpy.linalg import eig, norm
from numpy.typing import NDArray
from typing import Tuple
from tqdm import tqdm
from pandas import DataFrame

def prepare_dataframe_for_pca(indicators: DataFrame) -> NDArray:
    """
    Converts the dataframe of the merged indicators into a `numpy` array for PCA.

    The values are averaged. Any row with at least one column is dropped from the analysis at this
    stage.

    Args:
        - indicators: The merged dataset of the indicators.

    Returns: The indicators without any european agregate to which all rows with at least one null
        value is removed. The remaining rows are aggregated by country with an average of the value.
        This array is in a `numpy` format.
    """
    filtered_dataframe = indicators[~indicators.Country.str.contains('European')].dropna()
    filtered_dataframe = filtered_dataframe.groupby(['Country']).mean()
    return filtered_dataframe.iloc[:, 1:].to_numpy()

def apply_pca_on_indicators(
    indicators: NDArray
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Does the PCA on the indicators data given as a parameter.

    This method does, more precisely, a PCA on a correlation matrix wihtout any transformation.

    Args:
        - indicators: The data of the indicators. The data should be complete and have no empty
            data entry.

    Returns: The results of the PCA, in the form of a tuple. The first value of the tuple gives the
        eigen values. The second gives the eigen vectors and the last gives the explained variance
        by each principal component.
    """
    data = array(indicators)

    # From here, most of the code is taken from
    # https://bagheri365.github.io/blog/Principal-Component-Analysis-from-Scratch/
    # Variable names are changed.

    average = mean(data, axis=0)
    standard_deviation = std(data, axis=0)
    data = (data - average) / standard_deviation

    correlation_matrix = corrcoef(data.T)
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

def get_degrees_of_independance(eigen_vectors: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Computes the degrees of independances with the results of the PCA. `apply_pca_on_indicators`
    should be run beforehand.

    Args:
        - eigen_vectors: The eigen vectors from the principal component analysis.

    Returns: The results of the degrees of independance in the form of a tuple. The first element is
        the array of the angles between each indicator. The second is the arry of the degrees of
        independance. This degree is valued between 0 and 1. 0 means that two indicators are
        completely independant while 1 means that two indicators are completely dependant. Both of
        the matrices are triangular, so `j >= i` to have results. On the other cases, the value
        `0.0` has no meaning.
    """
    loading_vectors = eigen_vectors[:,:2]
    angle_matrix = array([[0.0] * len(loading_vectors)] * len(loading_vectors))
    for i, loading_vector in tqdm(
        enumerate(loading_vectors),
        "Computing angles between indicators",
        len(loading_vectors),
        leave=False
    ):
        for j in range(i+1, len(loading_vectors)):
            angle = rad2deg(
                    acos(
                        dot(loading_vector, loading_vectors[j])
                        / (norm(loading_vector) * norm(loading_vectors[j]))
                    )
            )

            angle_matrix[i][j] = angle

    independance_matrix = array(angle_matrix)
    independance_matrix[independance_matrix > 90] = 180 - independance_matrix[
        independance_matrix > 90
    ]
    independance_matrix = independance_matrix / 90

    return angle_matrix, independance_matrix
