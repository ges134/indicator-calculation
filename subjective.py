"""
This module simulates an AHP for the indicators that are studied. This gives the subjective weight
for the integrated objective-subjective method that is being developed. The weights are
determined by a comparison of indicators to each sustainability pillars.
"""

from typing import Dict, Tuple, List
from tqdm import tqdm
from numpy import full, where, iscomplex, real, average
from numpy.typing import NDArray
from numpy.linalg import eig
from numpy.random import default_rng
from pandas import DataFrame

PILLARS = ['social', 'environmental', 'economic']
PILLARS_COMPARISON_MATRIX = [
    [1, 1/9, 1/8],
    [9, 1, 1],
    [8, 1, 1]
]

def get_subjective_weights(comparison_matrices: Dict) -> Tuple[Dict, Dict, List[float]]:
    """
    From the comparison matrices, this method creates the weights for the subjective approach of
    the integrated objective-subjective method under development.

    This is done by AHP. This method focuses on the conversion of comparison matrices to weights.
    The matrices being converted are the social, economic, environemental and sustainability pillar
    matrices.

    Args:
        - comparison_matrices: A dictionnary containing the matrices of social, economic and
            environmental comparisons.

    Returns: A tuple containing the weights per sustainability pillars (with the weights of the
        pillars itself in another key), the consistency analysis per sustainbility pillars (with the
        weights of the pillars itself in another key) and the final weights.

        The weight vector is in the format of a dictionnary, vectors are inside the key of the
        corresponding pillar. For the weights of the pillars themselves, the key `'pillar'` can be
        used. The vector gives the weight of an indicator in an index basis. The keys associated to
        the indices must be computed outside this method.

        The consistency analysis dictionnary in which the consistency analysis is inside the key of
        the corresponding pillar. For the consistency analysis of the pillars themselves, the key 
        `'pillar'` can be used. The value is another dictionnary in which the key `'eigen_value'`
        gives the eigen value for this AHP, `'index'` gives the consistency index and `'ratio'`
        gives the consistency ratio.

        The final weights is in the format of an array. It gives the final weight of an indicator in
        an index basis. The indices must be computed outside this method.
    """

    weight_vectors = {
        'social': [],
        'economic': [],
        'environmental': [],
        'pillar': []
    }
    pillars_eigen_values = {
        'social': 0,
        'economic': 0,
        'environmental': 0,
        'pillar': 0
    }

    for pillar in tqdm(PILLARS, "Computing AHP - Weights", len(PILLARS), leave=False):
        eigen_value, normalized_weight_vector = get_weights_from_matrix(comparison_matrices[pillar])
        weight_vectors[pillar] = normalized_weight_vector
        pillars_eigen_values[pillar] = eigen_value

    pillars_eigen_value, pillars_weights = get_weights_from_matrix(PILLARS_COMPARISON_MATRIX)
    weight_vectors['pillar'] = pillars_weights
    pillars_eigen_values['pillar'] = pillars_eigen_value

    final_social_weights = weight_vectors['social'] * pillars_weights[0]
    final_economic_weights = weight_vectors['environmental'] * pillars_weights[1]
    final_environmental_weights = weight_vectors['economic'] * pillars_weights[2]

    final_weights = [
        social_weight + final_economic_weights[i] + final_environmental_weights[i] for i, social_weight in enumerate(final_social_weights)
    ]

    consistency = {
        'social': {
            'eigen_value': 0,
            'index': 0,
            'ratio': 0
        },
        'economic': {
            'eigen_value': 0,
            'index': 0,
            'ratio': 0
        },
        'environmental': {
            'eigen_value': 0,
            'index': 0,
            'ratio': 0
        },
        'pillar': {
            'eigen_value': 0,
            'index': 0,
            'ratio': 0
        }
    }

    for pillar in tqdm(pillars_eigen_values, "Computing AHP - consistency", 4, False):
        eigen_value = pillars_eigen_values[pillar]
        consistency[pillar]['eigen_value'] = eigen_value
        index, ratio = get_consistency_ratio(eigen_value, len(weight_vectors[pillar]))
        consistency[pillar]['index'] = index
        consistency[pillar]['ratio'] = ratio

    return weight_vectors, consistency, final_weights

def get_scores_for_indicators(config: Dict) -> Dict:
    """
    Converts the indicators score into a Likert scale to use for AHP.

    The indicators are scored between 0 and 3 with the following definitions for each sustainability
    pillar:
    - 0: The indicator does not cover this pillar.
    - 1: The indicator covers the pillar, but it is not a considered as the main focus of the
        indicator.
    - 2: The indicator covers the pillar and shares its main focus with another pillar.
    - 3: The indicator covers the pillar and its the main focus.

    The indicators score must be computed beforehand and must be provided in the configuration file.
    This method uses these score to create a Liket scale to use for AHP. Each pillar is associated
    with a score, 3 for the pillar and 1 for the other pillars. Then, a five point score is computed
    with the following rules:
    - 1: The indicator has a difference of at least two points with the pillar.
    - 3: The indicator has a difference of one point with the pillar.
    - 5: The indicator has the same score as the pillar. A difference of 3 or 4 point is observed
        for the other pillars.
    - 7: The indicator has the same score as the pillar. A difference of 1 or 2 point is observed
        for the other pillars.
    - 9: The indicator is a perfect match with the pillar.

    Args:
        - config: The configuration file for this program execution. It should provide an identifier
            and the pillars scores under the keys `'social'`, `'economic'` and `'environmental'`.
            The scores should be between 0 and 3.
    Returns:
        A dictionnary with the Likert scale. In this dictionnary, each indicator identifier
        represents the key. The value is another dictionnary to which each pillar is a key and the
        value is the Likert scale value for this pillar.
    """

    scores = {}

    for pillar in tqdm(PILLARS, "Computing AHP - Comparison scores", len(PILLARS), leave=False):
        social_score = 3 if pillar == 'social' else 1
        environmental_score = 3 if pillar == 'environmental' else 1
        economic_score = 3 if pillar == 'economic' else 1

        for indicator in config:
            indicator_score_for_pillar = 1
            if (
                indicator['social'] == social_score
                and indicator['environmental'] == environmental_score
                and indicator['economic'] == economic_score
            ):
                indicator_score_for_pillar = 9
            else:
                difference_social = abs(indicator['social'] - social_score)
                difference_environmental = abs(indicator['environmental'] - environmental_score)
                difference_economic = abs(indicator['economic'] - economic_score)
                if pillar == 'social':
                    if indicator['social'] == social_score:
                        indicator_score_for_pillar = 7 \
                            if difference_economic + difference_environmental < 3 \
                            else 5
                    elif difference_social < 2:
                        indicator_score_for_pillar = 3
                elif pillar == 'environmental':
                    if indicator['environmental'] == environmental_score:
                        indicator_score_for_pillar = 7 \
                            if difference_economic + difference_social < 3 \
                            else 5
                    elif difference_environmental < 2:
                        indicator_score_for_pillar = 3
                elif indicator['economic'] == economic_score:
                    indicator_score_for_pillar = 7 \
                        if difference_environmental + difference_social < 3 \
                        else 5
                elif difference_economic < 2:
                    indicator_score_for_pillar = 3

            if indicator['id'] not in scores:
                scores[indicator['id']] = {
                    'social': 0,
                    'environmental': 0,
                    'economic': 0
                }

            scores[indicator['id']][pillar] = indicator_score_for_pillar

    return scores

def get_comparison_matrices(scores: Dict) -> Dict:
    """
    Converts Likert scores into comparison matrices.

    This take the Likert scores and substract the values so that the lowest Likert score is one and
    the highest represents the difference. The comparison matrix is then built using AHP
    methodology.

    Args:
        - scores: The Likert scale of each indicator. This can be computed with the
            `get_scores_for_indicators` method.
    
    Returns: A dictionnary in which the pillars are key and the values are the comparison matrices.
        This is a square matrix of the size of the number of indicators.
    """

    comparison_matrices = {
        'social': [],
        'economic': [],
        'environmental': []
    }

    indicators = scores.keys()
    number_of_indicators = len(indicators)

    for pillar in tqdm(PILLARS, "Computing AHP - Comparison matrices", len(PILLARS), leave=False):
        comparison_matrix = full((number_of_indicators, number_of_indicators), 0.0)

        for i, indicator in enumerate(indicators):
            for j, other_indicator in enumerate(indicators):
                difference = scores[indicator][pillar] - scores[other_indicator][pillar]
                comparison_score = 1
                if difference > 0:
                    comparison_score = difference + 1
                elif difference < 0:
                    comparison_score = 1 / (abs(difference) + 1)

                comparison_matrix[i][j] = comparison_score

        comparison_matrices[pillar] = comparison_matrix

    return comparison_matrices

def get_weights_from_matrix(comparison_matrix: NDArray) -> Tuple[float, NDArray]:
    """
    Gets the weights and the associated eigen value from a given comparison matrix.

    Per AHP methodology, the maximal real eigen value, and its corresponding eigen vector gives the
    weights associated to a comparison matrix. These values are returned by the program.

    Args:
        - comparison_matrix: A comparison matrix. It should be a square matrix in which the diagonal
            is one and the values are reprocicated between the upper and the lower triangle. The
            method `get_comparison_matrices` can generate such matrices although it is not a
            prerequiste to run the method beforehand.

    Returns: A tuple in which the first value is the eigen value of the weights and the second value
        is the weights associated to that comparison matrix.
    """

    eigen_values, eigen_vectors = eig(comparison_matrix)
    real_values_indices = where(~iscomplex(eigen_values))[0]
    max_value = 0
    max_index = 0
    for i in real_values_indices:
        value = real(eigen_values[i])
        if value > max_value:
            max_value = value
            max_index = i

    weight_vector = real(eigen_vectors[:, max_index])
    normalized_weight_vector = weight_vector / sum(weight_vector)
    return max_value, normalized_weight_vector

def generate_random_comparison_matrix(size: int) -> NDArray:
    """
    Generates a random comparison matrix to compute the random index.

    The upper triangle of the random matrix is generated by drawing random values on the possibles 
    values, ranging from 1 to 9 and including their reciprocal. The lower triangle is the reciprocal
    of the lower triangle. The diagonal is set to one.

    Args:
        - size: The size of the random comparison matrix to generate.

    Returns: The generated comparison matrix.
    """

    values = [1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    matrix = full((size, size), 0.0)
    rng = default_rng()
    for i in range(size):
        for j in range(size):
            if i == j:
                matrix[i][j] = 1
            elif j < i:
                matrix[i][j] = 1 / matrix[j][i]
            else:
                matrix[i][j] = rng.choice(values)

    return matrix

def get_consistency_index(eigen_value: float, size: int) -> float:
    """
    Computes the consistency index for a comparison matrix of given size with given eigen value.

    Args:
        - eigen_value: The eigen value associated to this comparison matrix's weights.
        - size: The size of the comparison matrix.

    Returns: The consistency index, as described by AHP.
    """

    if size == 1:
        return 0.0
    return (eigen_value - size) / (size - 1)

def get_random_index(size: int) -> float:
    """
    Generates a random index for the specified matrix size.

    The random index is generated by following Donegan & Dodd's guidelines (1991). That is, 100
    random comparison matrices are generated with `generate_random_comparison_matrix`. Their
    consistencies are then averaged to give the random index.

    It is possible that the random index is different of the one reported by Saaty. The confidence
    interval is based on a 99% confidence level. This method was programmed since the usage will
    may require the computation of comparison matrices that have an order higher than 15, which is 
    the higest order reported by Saaty. As a comparison, the highest reported by Donegan & Dodd 
    (1991) is 20 before jumping units.

    Donegan & Dodd's study can be found with the following URL:
    https://www.sciencedirect.com/science/article/pii/089571779190098R

    Args:
        - size: The size of the random index.

    Returns: The generated random index.
    """

    consistency_indices = []
    for _ in tqdm(range(100), "Computing a random index", 100, False):
        matrix = generate_random_comparison_matrix(size)
        eigen_value, _ = get_weights_from_matrix(matrix)
        consistency_indices.append(get_consistency_index(eigen_value, size))

    return average(consistency_indices)

def get_consistency_ratio(eigen_value: float, size: int) -> Tuple[float, float]:
    """
    Computes the consistency ratio for a comparison matrix of given size with given eigen value.

    Args:
        - eigen_value: The eigen value associated to this comparison matrix's weights.
        - size: The size of the comparison matrix.

    Returns: The consistency ratio, as described by AHP.
    """

    consistency_index = get_consistency_index(eigen_value, size)
    random_index = get_random_index(size)
    return (consistency_index, consistency_index / random_index)

def convert_scores_to_dataframe(scores: Dict) -> DataFrame:
    """
    Converts the Likert scores into a Pandas `DataFrame` so that it can be saved into a file
    afterwards.

    Args:
        - scores: The Likert scores obtained through `get_scores_for_indicator`.

    Returns: the converted DataFrame.
    """

    indicators = scores.keys()
    scores_array = [
        [
            indicator,
            scores[indicator]['social'],
            scores[indicator]['economic'],
            scores[indicator]['environmental']
        ] for indicator in indicators
    ]
    columns = ['indicator', 'social', 'economic', 'environmental']
    return DataFrame(scores_array, columns=columns)

def convert_weights_to_dataframe(
    indicators: List[str],
    weight_vectors: Dict,
    final_weights: List
) -> DataFrame:
    """
    Converts the weights into a Pandas `DataFrame` so that it can be saved into a file afterwards.

    Args:
        - indicators: The list of indicators used in this program execution.
        - weight_vectors: The dictionnary of weights obtained through `get_subjective_weights`.
        - final_weights: The final weights of indicators obtained through `get_subjective_weights`.

    Returns: the converted DataFrame.
    """

    weight_array = [
        [
            indicator,
            weight_vectors['social'][i],
            weight_vectors['economic'][i],
            weight_vectors['environmental'][i],
            final_weights[i]
        ] for i, indicator in enumerate(indicators)
    ]
    columns = ['indicator', 'social', 'economic', 'environmental', 'final']
    return DataFrame(weight_array, columns=columns)

def convert_consistency_to_dataframe(consistency: Dict) -> DataFrame:
    """
    Converts the consistency analysis into a Pandas `DataFrame` so that it can be saved into a file 
    afterwards.

    Args:
        - consistency: The dictionnary with the consistency analysis

    Returns: the converted DataFrame.
    """

    consistency_array = [
        [
            pillar,
            consistency[pillar]['eigen_value'],
            consistency[pillar]['index'],
            consistency[pillar]['ratio']
        ] for pillar in consistency
    ]
    columns = ['pillar', 'eigen value', 'consistency index', 'consistency ratio']
    return DataFrame(consistency_array, columns=columns)
