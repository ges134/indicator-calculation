"""
This module provides automated tests for the `Independance` module.
"""

from unittest import TestCase
from unittest.mock import patch, Mock
from pyjstat.pyjstat import Dataset

from merger import merge_datasets, convert_dataset_to_dataframe
from independance import get_degrees_of_independance, prepare_dataframe_for_pca
from data import load_file
from stats import apply_pca
from tests.constants import DATA

CONFIG = [
    {
        'id': 'EMA',
        'code': 'cei_pc020'
    },
    {
        'id': 'PDR',
        'code': 'cei_pc030',
        'Unit of measure': 'Euro per kilogram, chain linked volumes (2015)'
    },
    {
        'id': 'GMR',
        'code': 'cei_pc034'
    }
]

class TestIndependance(TestCase):
    """
    This class provides automated tests for the module's methods.
    """

    @patch('merger.load_dataset')
    def test_prepare_dataframe_for_pca(self, load_dataset_mock: Mock):
        """
        Tests the method `prepare_dataframe_for_pca` under the normal scenario.
        """
        codes = ['cei_pc020', 'cei_pc030', 'cei_pc034', 'sdg_01_10', 'sdg_03_42']
        dataframes = []
        for code in codes:
            stub = Dataset.read(load_file(f'tests/{code}.json'))
            dataframes.append(stub.write('dataframe'))

        load_dataset_mock.side_effect = dataframes
        merged = merge_datasets(CONFIG)
        merged_dataframe = convert_dataset_to_dataframe(merged, CONFIG)
        expected_results = [
            [24.093625, 2.213175, 6855.25],
            [14.124, 2.8161, 5373.75],
            [17.961, 0.3377375, 19839.875],
            [14.182375, 1.074075, 1173.75],
            [23.1355, 1.20535, 2628.0],
            [17.285625, 1.039125, 2797.0],
            [22.8925, 2.0711875, 3313.625],
            [26.668375, 0.5979875, 15879.5],
            [49.04275, 0.9194625, 19492.0],
            [14.4055, 2.8328625, 5076.5],
            [15.934625, 2.450625, 4690.5],
            [15.171375, 1.2815, 5239.875],
            [12.222875, 0.9562, 1849.25],
            [17.6605, 2.394275, 3494.0],
            [11.925875, 3.0921125, 2828.125],
            [15.4478, 0.95207, 997.1],
            [18.221125, 0.815625, 2131.125],
            [32.2038, 4.04818, 16646.9],
            [12.277, 2.079075, 4694.375],
            [9.885375, 3.9822875, 7427.5],
            [15.721625, 0.673825, 4430.125],
            [17.50375, 1.080225, 1500.5],
            [22.434625, 0.398475, 9456.5],
            [16.146667, 0.325617, 6982.5],
            [14.9375, 1.16665, 2012.25],
            [18.5205, 1.388825, 3187.0],
            [11.640625, 2.3895, 2692.875],
            [25.231625, 1.9544375, 14216.0]
        ]

        # Act
        results = prepare_dataframe_for_pca(merged_dataframe)

        # Assert
        for i, row in enumerate(results):
            for j, value in enumerate(row):
                self.assertAlmostEqual(value, expected_results[i][j], 6)

    def test_get_degrees_of_independance(self):
        """
        Tests the method `get_degrees_of_independance` under the nominal scenario.
        """
        # Arrange
        _, eigen_vectors, _ = apply_pca(DATA)

        expected_angles = [
            [0, 118.7, 32.2, 115.9, 169.6, 89],
	        [0, 0, 150.9, 125.4, 71.7, 152],
            [0, 0, 0, 83.6, 137.3, 56.7],
            [0, 0, 0, 0, 53.7, 26.9],
            [0, 0, 0, 0, 0, 80.7],
            [0, 0, 0, 0, 0, 0]
        ]

        expected_independance = [
            [0, 0.681, 0.359, 0.712, 0.116, 0.989],
	        [0, 0, 0.323, 0.606, 0.797, 0.308],
            [0, 0, 0, 0.929, 0.474, 0.631],
            [0, 0, 0, 0, 0.596, 0.299],
            [0, 0, 0, 0, 0, 0.895],
            [0, 0, 0, 0, 0, 0]
        ]

        # Act
        angle_matrix, independance_matrix = get_degrees_of_independance(eigen_vectors)

        # Assert
        for i, row in enumerate(angle_matrix):
            for j, value in enumerate(row):
                self.assertAlmostEqual(value, expected_angles[i][j], 0)

        for i, row in enumerate(independance_matrix):
            for j, value in enumerate(row):
                self.assertAlmostEqual(value, expected_independance[i][j], 3)

        self.assertEqual((6, 6), angle_matrix.shape)
        self.assertEqual((6, 6), independance_matrix.shape)
