"""
This module computes the confidence intervals with the Bootstrap method. It consists of realizing
multiple bootstraped PCAs to compute the confidence intervals. The method is adapted from this
reference:

Babamoradi, H., van den Berg, F., & Rinnan, Å. (2013). Bootstrap based confidence limits in
principal component analysis—A case study. Chemometrics and Intelligent Laboratory Systems, 120,
97‑105. https://doi.org/10.1016/j.chemolab.2012.10.007
"""

from typing import List, Tuple
from pandas import DataFrame
from tqdm import tqdm
from numpy import (
    full,
    hstack,
    concatenate,
    abs,
    argmax,
    unique,
    mean,
    zeros,
    power,
    floor,
    sort,
    array
)
from numpy.typing import NDArray
from scipy.stats import Normal, norm

from stats import generate_bootstraped_dataset, apply_pca, correlation_matrix_between_pcas, jacknife

NUMBER_OF_SAMPLES = 2000

def generate_bootstraped_pcas_on_indicators(
    indicators: NDArray,
    empiric_eigen_vectors: NDArray
) -> Tuple[List[NDArray], List[NDArray]]:
    """
    Generates several Bootstrap samples and their associated PCAs to compute confidence intervals.

    Args:
        - indicators: The indicators to bootstrap.
        - empiric_eigen_vectors: The eigen vectors of the PCA applied to the `indicators` parameter.
            In the program, this is done beforehand.

    Returns: A tuple with two three-dimensional arrays. The first element of the tuple returns the
        bootstrap samples. In this array, the first index corresponds to the bootstrap sample
        number. The second index corresponds to the row (observation), and the third index returns
        the value of an indicator. The second element of the tuple returns the associated PCAs for
        the bootstraped dataset. In this array, the first index corresponds to the bootstrap sample
        number. The second index represents the indicator, and the third index represents the
        eigenvalue components for each principal component.
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
    This method ensures that the first principal component of the bootstraped PCA is equivalent to
    the first principal component of the empirical dataset, and so on.

    The axis reordering is based on the correlation matrix between the two PCAs. The strongest
    absolute correlation represents the association between the bootstraped PCA and the empirical
    PCA. This statement is only valid if all the mappings are distinct. If not, the sample is
    redrawn until the condition is proper.

    The axis reflection is performed by examining the correlation matrix after the axis reordering.
    If the sign of the correlation is negative, then the axis is multiplied by `-1`.

    Args:
        - indicators: The indicators to bootstrap.
        - empiric_eigen_vectors: The eigen vectors of the PCA applied to the `indicators` parameter.
            In the program, this is done beforehand.

    Returns: A tuple with two matrices. The first matrix is the Bootstrap sample, and the second is
        the PCA eigenvectors associated with this Bootstrap sample, with axis reflection and
        reordering.
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
    Jackknifes the data and then, for each jackknifed sample, applies the PCA.

    See `stats.jacknife` to understand the jacknifing procedure and `stats.apply_pca` to understand
    the PCA.

    Args:
        - indicators: The indicators to jackknife.

    Returns: A tuple with two three-dimensional arrays. The first element of the tuple returns the
        jacknifed samples. In this array, the first index corresponds to the row number of the
        removed element. The second index represents the row (observation), and the third index
        returns the value of an indicator. The second element of the tuple returns the associated
        PCAs for the jackknifed dataset. In this array, the first index corresponds to the
        jackknifed sample number. The second index represents the indicator, and the third index
        represents the components of the eigenvectors for each principal component.
    """

    jacknifed_data = jacknife(indicators)
    jacknifed_pca = []
    for sample in jacknifed_data:
        _, eigen_vectors, _ = apply_pca(sample)
        jacknifed_pca.append(eigen_vectors)

    return jacknifed_data, jacknifed_pca

def confidence_intervals_from_indexes(indexes: NDArray, bootstraped_pcas: NDArray) -> NDArray:
    """
    Gives the set of bounds from the given index.

    This method allows the index to be converted into a part of the confidence interval. It must be
    called twice, once with the indices of the lower bounds and again with the indices of the upper
    bounds. The bounds are located in the bootstraped estimations, which are sorted during the
    execution of the method.

    Args:
        - indexes: The set of indexes associated with the bounds.
        - bootstraped_pcas: The bootstrap estimations. The actual bounds will be returned from this
            parameter.

    Returns: An array with the bounds of a confidence interval.
    """

    confidence_intervals = []
    for i, row in enumerate(indexes):
        confidence_interval_row = []
        for j, index in enumerate(row):
            confidence_interval_row.append(sort(bootstraped_pcas[:, i, j])[index])
        confidence_intervals.append(confidence_interval_row)

    return array(confidence_intervals)

def produce_confidence_intervals(
    bootstraped_pcas: NDArray,
    jacknifed_pcas: NDArray,
    empiric_eigen_vectors: NDArray,
    significant_level: float
) -> Tuple[NDArray, NDArray]:
    """
    Computes the confidence intervals for the eigen vectors based on the specified significance
    level.

    This method requires that the bootstrap and jackknife datasets have been generated beforehand.
    Hence, the methods' bootstrap_and_apply_pca` and `jacknife_and_apply_pca` should have been
    called beforehand. Furthermore, the empirical eigenvectors are also required and can be
    generated using `stats.apply_pca`.

    Args:
        - bootstraped_pcas: The bootstrap estimations. The actual bounds will be returned from this
            parameter.
        - jacknifed_pcas: The PCAs of the jacknifed dataset.
        - empiric_eigen_vector: The eigen vectors of the PCA applied to the data in which the
            bootstraped dataset was generated.
        - significant_level: A value between 0 and 1 for the significant level. Recommended values
            are 0.05 and 0.01.

    Returns: A tuple containing the bounds of the confidence intervals. The first element of the
        tuple returns the lower bounds, while the second returns the upper bounds.   
    """

    standard_normal_distribution = Normal(mu=0, sigma=1)

    bootstraped_means = mean(bootstraped_pcas, axis=0)
    number_below_empiric = zeros(bootstraped_means.shape)
    for sample in bootstraped_pcas:
        for i, row in enumerate(sample):
            for j, element in enumerate(row):
                if element < empiric_eigen_vectors[i, j]:
                    number_below_empiric[i, j] += 1
    proportion_below_empiric = number_below_empiric / len(bootstraped_pcas)
    bias_corrector = standard_normal_distribution.icdf(proportion_below_empiric)

    jacknifed_means = mean(jacknifed_pcas, axis=0)
    distribution_skewness = sum(power(jacknifed_pcas - jacknifed_means, 3)) \
        / (6 * power(sum(power(jacknifed_pcas - jacknifed_means, 2)), 3/2))

    significant_level_lower_critical_value = norm.ppf(significant_level / 2)
    significant_level_upper_critical_value = norm.ppf(1 - (significant_level / 2))
    lower_confidence_location = standard_normal_distribution.cdf(
        bias_corrector + (
            (
                bias_corrector + significant_level_lower_critical_value
            ) / (
                1 - distribution_skewness * (
                    bias_corrector + significant_level_lower_critical_value
                )
            )
        )
    )
    upper_confidence_location = standard_normal_distribution.cdf(
        bias_corrector + (
            (
                bias_corrector + significant_level_upper_critical_value
            ) / (
                1 - distribution_skewness * (
                    bias_corrector + significant_level_upper_critical_value
                )
            )
        )
    )
    lower_confidence_index = floor(lower_confidence_location * len(bootstraped_pcas)).astype(int)
    upper_confidence_index = floor(upper_confidence_location * len(bootstraped_pcas)).astype(int)
    lower_confidence_intervals = confidence_intervals_from_indexes(
        lower_confidence_index,
        bootstraped_pcas
    )
    upper_confidence_intervals = confidence_intervals_from_indexes(
        upper_confidence_index,
        bootstraped_pcas
    )

    return (lower_confidence_intervals, upper_confidence_intervals)

def bootstraped_indicators_to_dataframe(
    bootstraped_indicators: List[NDArray],
    codes: List[str]
) -> DataFrame:
    """
    Converts the bootstraped indicators into a dataframe for later saving as a CSV file.
    
    Args:
        - bootstraped_indicators: The bootstraped indicators generated by
            `generate_bootstraped_pcas_on_indicators`. The three-dimensional array will be
            flattened into a two-dimensional array.
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
    Converts the jacknifed indicators into a dataframe for later saving as a CSV file.
    
    Args:
        - jacknifed_indicators: The jacknifed indicators generated by `jacknife`.
            The three-dimensional array will be flattened to a two-dimensional array.
        - codes: The list of indicator identifiers to show in the header of the dataframe.

    Return: The converted dataframe.
    """

    flattened_indicators = flatten(jacknifed_indicators)
    columns = ['Jacknife number'] + codes
    return DataFrame(flattened_indicators, columns=columns)

def confidence_interval_to_dataframe(
    lower_confidence_interval: NDArray,
    upper_confidence_interval: NDArray,
    codes: List[str]
) -> DataFrame:
    """
    Converts the confidence intervals into a dataframe for later saving as a CSV file.

    Args:
        - lower_confidence_interval: The lower bounds of the confidence intervals.
        - upper_confidence_interval: The upper bounds of the confidence intervals.

    Return: The converted dataframe.
    """

    dataframe_data = []
    for i, row in enumerate(lower_confidence_interval):
        dataframe_data.append([codes[i], 'lb'] + row.tolist())
    for i, row in enumerate(upper_confidence_interval):
        dataframe_data.append([codes[i], 'ub'] + row.tolist())

    columns = ['indicator', 'confidence interval bound'] \
        + [f'PC {i + 1}' for i in range(len(lower_confidence_interval[0]))]

    return DataFrame(dataframe_data, columns=columns)

def flatten(data: List[NDArray]) -> NDArray:
    """
    Flattens a three-dimensional array into a two-dimensional array. The index of the third
    dimension is added as a column to the two-dimensional array.

    Args:
        - data: The data to convert into a two-dimensional array.

    Return: The flattened array. The first column of this array corresponds to the index of the
        third dimension.
    """
    flattened: NDArray = None
    for i, sample in tqdm(enumerate(data), "Flattening data.", leave=False):
        index_column = full((len(data[i]), 1), i + 1)
        entries = hstack((index_column, sample))
        flattened = entries if flattened is None else concatenate((flattened, entries))

    return flattened
