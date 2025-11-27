"""
This module computes the degree of independence from the indicators. The degrees follow the
interpretation of principal component analysis. The closer an indicator is to 90 degrees from the
eigen vector of the first two principal components, the more independent it is.
"""

from numpy import array, rad2deg, acos, dot
from numpy.linalg import norm
from numpy.typing import NDArray
from typing import Tuple
from tqdm import tqdm
from pandas import DataFrame

def prepare_dataframe_for_pca(indicators: DataFrame) -> NDArray:
    """
    Converts the merged indicators dataframe into a `numpy` array for PCA.

    The values are averaged. Any row with at least one column with a `nan` is dropped from the
    analysis at this stage.

    Args:
        - indicators: The merged dataset of the indicators.

    Returns: The indicators without any European aggregate to which all rows with at least one null
        value are removed. The remaining rows are aggregated by country, with the average value.
        This array is in a `numpy` format.
    """
    filtered_dataframe = indicators[~indicators.Country.str.contains('European')].dropna()
    filtered_dataframe = filtered_dataframe.groupby(['Country']).mean()
    return filtered_dataframe.iloc[:, 1:].to_numpy()

def get_pca_data_from_years(indicators: DataFrame, year: int) -> NDArray:
    """
    Converts a dataframe of merged indicators into a `numpy` array of PCA with all the observations
    of a given year.

    Any row with at least one column with a `nan` is dropped from the analysis at this stage.

    Args:
        - indicators: The merged dataset of the indicators.
        - year: The year to retrieve the indicators from.

    Returns: The indicators without any European aggregate to which all rows with at least one null
        value are removed. The remaining rows represent one year, as specified by the `year`
        argument. This array is in a `numpy` format.
    """

    filtered_dataframe = indicators[~indicators.Country.str.contains('European')].dropna()
    filtered_dataframe = filtered_dataframe[filtered_dataframe.Year == year]
    return filtered_dataframe.iloc[:, 2:].to_numpy()

def get_degrees_of_independance(eigen_vectors: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Computes the degrees of independence with the results of the PCA. `stats.apply_pca`
    should be run beforehand.

    Args:
        - eigen_vectors: The eigen vectors from the principal component analysis.

    Returns: The results of the degrees of independence in the form of a tuple. The first element is
        the array of angles between each indicator. The second is the array of degrees of
        independence. This degree ranges from 0 to 1. 0 indicates that two indicators are entirely
        dependent, while 1 indicates that they are entirely independent. Both of the matrices are
        triangular, so `j >= i` to have results. In the other cases, the value `0.0` has no meaning.
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
