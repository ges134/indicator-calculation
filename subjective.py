"""
This module simulates an AHP for the indicators that are studied. This gives the subjective weight
for the integrated objective-subjective method that is being developed. The weights are
determined by a comparison of indicators to each sustainability pillars.
"""

from typing import Dict, Tuple, List
from tqdm import tqdm
from numpy import full, where, iscomplex, real
from numpy.typing import NDArray
from numpy.linalg import eig
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
        pillars itself in another key), the eigen values per sustainbility pillars (with the weights
        of the pillars itself in another key) and the final weights.

        The weight vector is in the format of a dictionnary, vectors are inside the key of the
        corresponding pillar. For the weights of the pillars themselves, the key `'pillars'` can be
        used. The vector gives the weight of an indicator in an index basis. The keys associated to
        the indices must be computed outside this method.

        The pillars eigen values is in the format of a dictionnary. The keys are the same as of the
        weights vectors. Instead of returning an array of weights, this dictionnary returns a single
        value.

        The final weights is in the format of an array. It gives the final weight of an indicator in
        an index basis. The indices must be computed outside this method.
    """

    weight_vectors = {
        'social': [],
        'economic': [],
        'environmental': [],
        'pillars': []
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
    weight_vectors['pillars'] = pillars_weights
    pillars_eigen_values['pillar'] = pillars_eigen_value

    final_social_weights = weight_vectors['social'] * pillars_weights[0]
    final_economic_weights = weight_vectors['environmental'] * pillars_weights[1]
    final_environmental_weights = weight_vectors['economic'] * pillars_weights[2]

    final_weights = [
        social_weight + final_economic_weights[i] + final_environmental_weights[i] for i, social_weight in enumerate(final_social_weights)
    ]

    return weight_vectors, pillars_eigen_values, final_weights

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
