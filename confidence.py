"""
This modules computes the confidence intervals with the Bootstrap method. It consists on realizing
multiple bootstraped PCAs to compute the confidence intervals. The method is adapted fromn this
reference:

Babamoradi, H., van den Berg, F., & Rinnan, Å. (2013). Bootstrap based confidence limits in
principal component analysis—A case study. Chemometrics and Intelligent Laboratory Systems, 120,
97‑105. https://doi.org/10.1016/j.chemolab.2012.10.007
"""

from typing import List, Tuple
from pandas import DataFrame
from tqdm import tqdm
from numpy import full, hstack, concatenate, abs, argmax, unique
from numpy.typing import NDArray

from stats import generate_bootstraped_dataset, apply_pca, correlation_matrix_between_pcas, jacknife

NUMBER_OF_SAMPLES = 2000

def generate_bootstraped_pcas_on_indicators(
    indicators: NDArray,
    empiric_eigen_vectors: NDArray
) -> Tuple[List[NDArray], List[NDArray]]:
    """
    Generates a numper of bootstrap samples and their associated PCAs to perform the computation of
    confidence intervals.

    Args:
        - indicators: The indicators to bootstrap.
        - empiric_eigen_vectors: The eigen vectors of the PCA applied to the `indicators` parameter.
            In the program, this is done beforehand.

    Returns: A tuple with two three dimensionnal array. The first element of the tuple returns the
        bootstrap samples. In this array, the first index represents the number of the bootstrap
        sample. The second index represents the row (observation) and the third index returns the
        value of an indicator. The second element of the tuple returns the associated PCAs for the
        bootstraped dataset. In this array, the first index represents the number of the bootstrap
        sample. The second index represents the indicator and the third index represents the
        eigen vectors components for each principal component.
    """
    bootstraped_pcas = []
    bootstraped_data = []

    for _ in tqdm(
        range(NUMBER_OF_SAMPLES),
        'Applying PCA with axis reordering and reversal on bootstraped data',
        leave=False
    ):
        final_bootstrap_sample, final_bootstraped_eigen_vectors = bootstrap_and_apply_pca(
            indicators,
            empiric_eigen_vectors
        )
        bootstraped_pcas.append(final_bootstraped_eigen_vectors)
        bootstraped_data.append(final_bootstrap_sample)

    return (bootstraped_data, bootstraped_pcas)

def bootstrap_and_apply_pca(
    indicators: NDArray,
    empiric_eigen_vectors: NDArray
) -> Tuple[NDArray, NDArray]:
    """
    Draws a bootstrap sample and applies a PCA on this sample with axis reordering and reversal.
    This method ensures that the first principal component of the bootstraped PCA is the equivalent
    of the first principal component of the empirical dataset and so on.

    The axis reordering is done by looking at the correlation matrix between the two PCAs. The
    strongest absolute correlation represents the association between the bootstraped PCA and the
    empirical PCA. This is only true if all the mappings are distinct. If not, the sample is
    redrawn until the condition is true.

    The axis reflexion is done by looking at the correlation matrix after the axis reordering. If
    the sign of the correlation is negative, then the axis is multiplied by `-1`.

    Args:
        - indicators: The indicators to bootstrap.
        - empiric_eigen_vectors: The eigen vectors of the PCA applied to the `indicators` parameter.
            In the program, this is done beforehand.

    Rerturns: A tuple with two matrices. The first matrix is the bootstraped sample and in the
        second is the eigen vectors of the PCA associated to this bootstraped example with axis
        reflection and reordering.
    """
    can_reorder = False
    final_bootstraped_eigen_vectors = []
    final_correlation_indexes = []
    final_correlation_matrix = []
    final_bootstrap_sample = []

    # Draw until we can reorder the data.
    while not can_reorder:
        bootstrap_sample = generate_bootstraped_dataset(indicators)
        _, bootstraped_eigen_vectors, _ = apply_pca(bootstrap_sample)
        correlation = correlation_matrix_between_pcas(
            empiric_eigen_vectors,
            bootstraped_eigen_vectors
        )
        absolute_correlation = abs(correlation)
        max_correlation = argmax(absolute_correlation, axis=0)
        can_reorder = len(bootstraped_eigen_vectors[0]) == len(unique(max_correlation))
        if can_reorder:
            final_bootstraped_eigen_vectors = bootstraped_eigen_vectors
            final_correlation_indexes = max_correlation
            final_bootstrap_sample = bootstrap_sample
            final_correlation_matrix = correlation

    original_axis = list(range(len(final_bootstraped_eigen_vectors[0])))
    final_bootstraped_eigen_vectors = final_bootstraped_eigen_vectors[:, final_correlation_indexes]
    final_correlation_matrix = final_correlation_matrix[:, final_correlation_indexes]

    for index in original_axis:
        if final_correlation_matrix[index][index] < 0:
            final_bootstraped_eigen_vectors[index, :] = final_bootstraped_eigen_vectors[index, :] * -1

    return final_bootstrap_sample, final_bootstraped_eigen_vectors

def jacknife_and_apply_pca(indicators: NDArray) -> Tuple[List[NDArray], List[NDArray]]:
    """
    Jacknifes the data and then, for each jacknifed sample, applies the PCA.

    See `stats.jacknife` to understand the jacknifing procedure and `stats.apply_pca` to understand
    the PCA.

    Args:
        - indicators: The indicators to jacknife.

    Returns: A tuple with two three dimensionnal array. The first element of the tuple returns the
        jacknifed samples. In this array, the first index represents the number of the row with a
        removed element. The second index represents the row (observation) and the third index 
        returns the value of an indicator. The second element of the tuple returns the associated 
        PCAs for the jacknifed dataset. In this array, the first index represents the number of the
        jacknifed sample. The second index represents the indicator and the third index represents
        the eigen vectors components for each principal component.
    """

    jacknifed_data = jacknife(indicators)
    jacknifed_pca = []
    for sample in jacknifed_data:
        _, eigen_vectors, _ = apply_pca(sample)
        jacknifed_pca.append(eigen_vectors)

    return jacknifed_data, jacknifed_pca

def bootstraped_indicators_to_dataframe(
    bootstraped_indicators: List[NDArray],
    codes: List[str]
) -> DataFrame:
    """
    Converts the bootstraped indicators into a dataframe to be later saved into a CSV file.
    
    Args:
        - bootstraped_indicators: The bootstraped indicators generated by
            `generate_bootstraped_pcas_on_indicators`. The three dimensional array will be flattened
            to a two dimension array.
        - codes: The list of indicator identifiers to show in the header of the dataframe.

    Return: The converted dataframe.
    """

    flattened_indicators = flatten(bootstraped_indicators)
    columns = ['Bootstrap sample number'] + codes
    return DataFrame(flattened_indicators, columns=columns)

def jacknifed_indicators_to_dataframe(
    jacknifed_indicators: List[NDArray],
    codes: List[str]
) -> DataFrame:
    """
    Converts the jacknifed indicators into a dataframe to be later saved into a CSV file.
    
    Args:
        - jacknifed_indicators: The jacknifed indicators generated by `jacknife`.
            The three dimensional array will be flattened to a two dimension array.
        - codes: The list of indicator identifiers to show in the header of the dataframe.

    Return: The converted dataframe.
    """

    flattened_indicators = flatten(jacknifed_indicators)
    columns = ['Jacknife number'] + codes
    return DataFrame(flattened_indicators, columns=columns)

def flatten(data: List[NDArray]) -> NDArray:
    """
    Flattens a three-dimensional array into a two-dimensional array. The index of the third
        dimension is added as a row to the two dimension array.

    Args:
        - data: The data to convert in a two-dimensional array.

    Return: The flattened array. The first column of this array correspond to the index of the third
        dimension.
    """
    flattened: NDArray = None
    for i, sample in tqdm(enumerate(data), "Flattening data.", leave=False):
        index_column = full((len(data[i]), 1), i + 1)
        entries = hstack((index_column, sample))
        flattened = entries if flattened is None else concatenate((flattened, entries))

    return flattened
