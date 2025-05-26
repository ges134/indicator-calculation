"""
This module provides automated tests for the `Subjective` module.
"""

from unittest import TestCase
from numpy import sqrt

from subjective import (
    get_scores_for_indicators,
    get_comparison_matrices,
    get_subjective_weights,
    convert_scores_to_dataframe,
    convert_weights_to_dataframe,
    convert_consistency_to_dataframe,
    get_random_index
)

CONFIG = [
  {
    'id': 'EMA',
    'code': 'cei_pc020',
    'social': 0,
    'environmental': 1,
    'economic': 3
  },
  {
    'id': 'PDR',
    'code': 'cei_pc030',
    'Unit of measure': 'Euro per kilogram, chain linked volumes (2015)',
    'social': 0,
    'environmental': 3,
    'economic': 1
  },
  {
    'id': 'GMR',
    'code': 'cei_pc034',
    'social': 2,
    'environmental': 2,
    'economic': 1
  },
  {
    'id': 'TRP',
    'code': 'sdg_01_10',
    'Age class': 'Total',
    'Unit of measure': 'Thousand persons',
    'social': 3,
    'environmental': 0,
    'economic': 1
  },
  {
    'id': 'NDE',
    'code': 'sdg_06_40',
    'social': 3,
    'environmental': 1,
    'economic': 0
  },
  {
    'id': 'PMS',
    'code': 'sdg_03_42',
    'Type of mortality': 'Preventable mortality',
    'social': 3,
    'environmental': 0,
    'economic': 0
  }
]

class TestSubjective(TestCase):
    """
    This class provides automated tests for the module's methods.
    """

    def test_get_scores_for_indicators(self):
        """
        Tests the method `get_scores_for_indicators` under the normal scenario.
        """

        # Arrange
        expected_dictionnary = {
            'EMA': {
                'social': 1,
                'environmental': 1,
                'economic': 7
            },
            'PDR': {
                'social': 1,
                'environmental': 7,
                'economic': 1
            },
            'GMR': {
                'social': 3,
                'environmental': 3,
                'economic': 1
            },
            'TRP': {
                'social': 7,
                'environmental': 1,
                'economic': 1,
            },
            'NDE': {
                'social': 7,
                'environmental': 1,
                'economic': 1,
            },
            'PMS': {
                'social': 7,
                'environmental': 1,
                'economic': 1,
            }
        }

        # Act
        results = get_scores_for_indicators(CONFIG)

        # Assert
        self.assertDictEqual(expected_dictionnary, results)

    def test_get_comparison_matrices(self):
        """
        Tests the method `get_comparison_matrices` under the normal scenario.
        """

        # Arrange
        expected_social_matrix = [
            [1, 1, 1/3, 1/7, 1/7, 1/7],
            [1, 1, 1/3, 1/7, 1/7, 1/7],
            [3, 3, 1, 1/5, 1/5, 1/5],
            [7, 7, 5, 1, 1, 1],
            [7, 7, 5, 1, 1, 1],
            [7, 7, 5, 1, 1, 1]
        ]
        expected_environmental_matrix = [
            [1, 1/7, 1/3, 1, 1, 1],
            [7, 1, 5, 7, 7, 7],
            [3, 1/5, 1, 3, 3, 3],
            [1, 1/7, 1/3, 1, 1, 1],
            [1, 1/7, 1/3, 1, 1, 1],
            [1, 1/7, 1/3, 1, 1, 1]
        ]
        expected_economie_matrix = [
            [1, 7, 7, 7, 7, 7],
            [1/7, 1, 1, 1, 1, 1],
            [1/7, 1, 1, 1, 1, 1],
            [1/7, 1, 1, 1, 1, 1],
            [1/7, 1, 1, 1, 1, 1],
            [1/7, 1, 1, 1, 1, 1]
        ]
        scores = get_scores_for_indicators(CONFIG)

        # Act
        matrices = get_comparison_matrices(scores)

        # Assert
        for i, row in enumerate(expected_social_matrix):
            self.assertListEqual(row, matrices['social'][i].tolist())

        for i, row in enumerate(expected_environmental_matrix):
            self.assertListEqual(row, matrices['environmental'][i].tolist())

        for i, row in enumerate(expected_economie_matrix):
            self.assertListEqual(row, matrices['economic'][i].tolist())

    def test_get_subjective_weights(self):
        """
        Tests the method `get_subjective_weights` under the normal scenario.
        """

        # Arrange
        expected_social_weights = [0.0359, 0.0359, 0.0757, 0.2842, 0.2842, 0.2842]
        expected_environmental_weights = [0.0672, 0.5501, 0.1810, 0.0672, 0.0672, 0.0672]
        expected_economic_weights = [0.5833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833]
        expected_pillars_weights = [0.0556, 0.4814, 0.4629]
        expected_eigen_values = {
            'social': 6.098,
            'economic': 6.0,
            'environmental': 6.0662,
            'pillar': 3.002
        }
        expected_final_weights = [0.3044, 0.3054, 0.1299, 0.0868, 0.0868, 0.0868]

        scores = get_scores_for_indicators(CONFIG)
        matrices = get_comparison_matrices(scores)

        standard_deviation = sqrt(10) * 0.0389
        pillar_standard_deviation = sqrt(10) * 0.0676
        random_index_lower_bound = 1.1797 - (standard_deviation * 3)
        random_index_upper_bound = 1.1797 + (standard_deviation * 3)
        # 3 goes to negative here.
        random_pillar_index_lower_bound = 0.4887 - (pillar_standard_deviation * 2)
        random_pillar_index_upper_bound = 0.4887 + (pillar_standard_deviation * 2)
        social_ratio_lower_bound = \
            ((expected_eigen_values['social'] - 6) / 5) / random_index_upper_bound
        social_ratio_upper_bound = \
            ((expected_eigen_values['social'] - 6) / 5) / random_index_lower_bound
        economic_ratio_lower_bound = \
            ((expected_eigen_values['economic'] - 6) / 5) / random_index_upper_bound
        economic_ratio_upper_bound = \
            ((expected_eigen_values['economic'] - 6) / 5) / random_index_lower_bound
        environmental_ratio_lower_bound = \
            ((expected_eigen_values['environmental'] - 6) / 5) / random_index_upper_bound
        environmental_ratio_upper_bound = \
            ((expected_eigen_values['environmental'] - 6) / 5) / random_index_lower_bound
        pillar_ratio_lower_bound = \
            ((expected_eigen_values['pillar'] - 3) / 2) / random_pillar_index_upper_bound
        pillar_ratio_upper_bound = \
            ((expected_eigen_values['pillar'] - 3) / 2) / random_pillar_index_lower_bound

        # Act
        weight_vectors, consistency, final_weights = get_subjective_weights(matrices)

        # Assert
        for i, weight in enumerate(expected_social_weights):
            self.assertAlmostEqual(weight, weight_vectors['social'][i], 4)

        for i, weight in enumerate(expected_environmental_weights):
            self.assertAlmostEqual(weight, weight_vectors['environmental'][i], 4)

        for i, weight in enumerate(expected_economic_weights):
            self.assertAlmostEqual(weight, weight_vectors['economic'][i], 4)

        for i, weight in enumerate(expected_pillars_weights):
            self.assertAlmostEqual(weight, weight_vectors['pillar'][i], 4)

        self.assertAlmostEqual(
            expected_eigen_values['social'],
            consistency['social']['eigen_value'],
            3
        )
        self.assertAlmostEqual(
            (expected_eigen_values['social'] - 6) / 5,
            consistency['social']['index'],
            3
        )
        self.assertTrue(
            social_ratio_lower_bound <= consistency['social']['ratio'] <= social_ratio_upper_bound
        )

        self.assertAlmostEqual(
            expected_eigen_values['economic'],
            consistency['economic']['eigen_value'],
            3
        )
        self.assertAlmostEqual(
            (expected_eigen_values['economic'] - 6) / 5,
            consistency['economic']['index'],
            3
        )
        self.assertTrue(
            economic_ratio_lower_bound \
                <= float(format(consistency['economic']['ratio'], '.4f')) \
                <= economic_ratio_upper_bound
        )

        self.assertAlmostEqual(
            expected_eigen_values['environmental'],
            consistency['environmental']['eigen_value'],
            3
        )
        self.assertAlmostEqual(
            (expected_eigen_values['environmental'] - 6) / 5,
            consistency['environmental']['index'],
            3
        )
        self.assertTrue(
            environmental_ratio_lower_bound \
                <= consistency['environmental']['ratio'] \
                <= environmental_ratio_upper_bound
        )

        self.assertAlmostEqual(
            expected_eigen_values['pillar'],
            consistency['pillar']['eigen_value'],
            3
        )
        self.assertAlmostEqual(
            (expected_eigen_values['pillar'] - 3) / 2,
            consistency['pillar']['index'],
            3
        )
        self.assertTrue(
            pillar_ratio_lower_bound <= consistency['pillar']['ratio'] <= pillar_ratio_upper_bound
        )

        for i, weight in enumerate(expected_final_weights):
            self.assertAlmostEqual(weight, final_weights[i], 4)

    def test_convert_scores_to_dataframe(self):
        """
        Tests the method `convert_scores_to_dataframe` under the normal scenario.
        """

        # Arrange
        scores = get_scores_for_indicators(CONFIG)

        # Act
        dataframe = convert_scores_to_dataframe(scores)

        # Assert
        for row in dataframe.itertuples():
            self.assertTrue(row.indicator in scores)
            self.assertEqual(scores[row.indicator]['social'], row.social)
            self.assertEqual(scores[row.indicator]['economic'], row.economic)
            self.assertEqual(scores[row.indicator]['environmental'], row.environmental)

    def test_convert_weihts_to_datafgrame(self):
        """
        Tests the method `get_convert_weihts_to_datafgrame` under the normal scenario.
        """
        # Arrange
        scores = get_scores_for_indicators(CONFIG)
        matrices = get_comparison_matrices(scores)
        weight_vectors, _, final_weights = get_subjective_weights(matrices)
        indicators = scores.keys()

        # Act
        dataframe = convert_weights_to_dataframe(indicators, weight_vectors, final_weights)

        # Assert
        for i, row in enumerate(dataframe.itertuples()):
            self.assertTrue(row.indicator in scores)
            self.assertEqual(weight_vectors['social'][i], row.social)
            self.assertEqual(weight_vectors['economic'][i], row.economic)
            self.assertEqual(weight_vectors['environmental'][i], row.environmental)
            self.assertEqual(final_weights[i], row.final)

    def test_convert_consistency_to_dataframe(self):
        """
        Tests the method `convert_consistency_to_dataframe` under the normal scenario.
        """

        # Arrange
        scores = get_scores_for_indicators(CONFIG)
        matrices = get_comparison_matrices(scores)
        _, consistency, _ = get_subjective_weights(matrices)

        # Act
        dataframe = convert_consistency_to_dataframe(consistency)

        # Assert
        for row in dataframe.itertuples():
            pillar = row.pillar
            self.assertTrue(pillar in consistency)
            self.assertEqual(consistency[pillar]['eigen_value'], row._2)
            self.assertEqual(consistency[pillar]['index'], row._3)
            self.assertEqual(consistency[pillar]['ratio'], row._4)

    def test_get_random_index(self):
        """
        Tests the method `test_get_random_index` under the normal scenario.
        """

        # Arrange
        # Taken from Donegan-Dodd
        # (https://www.sciencedirect.com/science/article/pii/089571779190098R)
        # Tuple: (size, average, std error)
        test_values = [
            (1, 0.0, 0.0),
            (2, 0.0, 0.0),
            (3, 0.4887, 0.0676),
            (4, 0.8045, 0.0609),
            (5, 1.0591, 0.0484),
            (6, 1.1797, 0.0389),
            (7, 1.2519, 0.0312),
            (8, 1.3171, 0.0267),
            (9, 1.3733, 0.0235),
            (10, 1.4055, 0.0215),
            (11, 1.4213, 0.0187),
            (12, 1.4497, 0.0165),
            (13, 1.4643, 0.0151),
            (14, 1.4822, 0.0139),
            (16, 1.4969, 0.0127),
        ]

        # Act
        results = [get_random_index(size) for size, _, _ in test_values]

        # Assert
        for i, random_index in enumerate(results):
            _, average, standard_error = test_values[i]
            rounded_random_index = float(format(random_index, '.4f'))
            standard_deviation = sqrt(10) * standard_error
            # This should be validated.
            lower_bound = average - (standard_deviation * 3)
            upper_bound = average + (standard_deviation * 3)
            self.assertTrue(lower_bound <= rounded_random_index <= upper_bound)
