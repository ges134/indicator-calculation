"""
This module provides automated tests for the `Subjective` module.
"""

from unittest import TestCase

from subjective import (
    get_scores_for_indicators,
    get_comparison_matrices,
    get_subjective_weights,
    convert_scores_to_dataframe,
    convert_weights_to_dataframe
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

        # Act
        weight_vectors, pillars_eigen_values, final_weights = get_subjective_weights(matrices)

        # Assert
        for i, weight in enumerate(expected_social_weights):
            self.assertAlmostEqual(weight, weight_vectors['social'][i], 4)

        for i, weight in enumerate(expected_environmental_weights):
            self.assertAlmostEqual(weight, weight_vectors['environmental'][i], 4)

        for i, weight in enumerate(expected_economic_weights):
            self.assertAlmostEqual(weight, weight_vectors['economic'][i], 4)

        for i, weight in enumerate(expected_pillars_weights):
            self.assertAlmostEqual(weight, weight_vectors['pillars'][i], 4)

        self.assertAlmostEqual(expected_eigen_values['social'], pillars_eigen_values['social'], 3)
        self.assertAlmostEqual(
            expected_eigen_values['economic'],
            pillars_eigen_values['economic'],
            3
        )
        self.assertAlmostEqual(
            expected_eigen_values['environmental'],
            pillars_eigen_values['environmental'],
            3
        )
        self.assertAlmostEqual(expected_eigen_values['pillar'], pillars_eigen_values['pillar'], 3)

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
