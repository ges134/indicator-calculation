"""
This module provides automated tests for the `Confidence` module.
"""

from unittest import TestCase
from unittest.mock import patch, Mock
from numpy import array, allclose
from numpy.typing import NDArray
from confidence import (
    NUMBER_OF_SAMPLES,
    bootstraped_indicators_to_dataframe,
    jacknifed_indicators_to_dataframe,
    bootstrap_and_apply_pca,
    generate_bootstraped_pcas_on_indicators,
    jacknife_and_apply_pca,
    confidence_intervals_from_indexes,
    produce_confidence_intervals,
    confidence_interval_to_dataframe
)
from stats import jacknife, apply_pca
from data import load_file
from tests.constants import DATA, LOWER_BOUNDS, UPPER_BOUNDS, EIGEN_VECTORS

CODES = ['cei_pc020', 'cei_pc030', 'cei_pc034', 'sdg_01_10', 'sdg_06_40', 'sdg_03_42']

class TestConfidence(TestCase):
    """
    This class provides automated tests for the module's methods.
    """

    bootstraped_pcas: NDArray
    jacknifed_pcas: NDArray

    def setUp(self):
        jacknifed_pcas_raw_data = load_file('tests/jacknifed-expected.csv')
        data = [
            [float(r) for r in l.split(',')[2:]] for l in jacknifed_pcas_raw_data.splitlines()
        ]
        jacknifed_pcas_data = []
        for i in range(0, len(data), 6):
            jacknifed_pcas_data.append(
                data[i:i+6]
            )
        self.jacknifed_pcas = array(jacknifed_pcas_data)

        bootstraped_raw_data = load_file('tests/bootstraped.csv')
        data = [
            [float(r) for r in l.split(',')] for l in bootstraped_raw_data.splitlines()
        ]
        bootstraped_data = []
        for i in range(0, len(data), 6):
            bootstraped_data.append(
                data[i:i+6]
            )
        self.bootstraped_pcas = array(bootstraped_data)

    @patch('confidence.generate_bootstraped_dataset')
    def test_bootstrap_and_apply_pca(self, generate_bootstraped_dataset_mock: Mock):
        """
        Tests the method `bootstrap_and_apply_pca` under the normal scenario.
        """

        # Arrange
        stub_bootstraped = array([
            [16.36666667, 106.1196667, 1132, 27.23333333, 3.816666667, 519.9066667],
            [27.29866667, 106.2463333, 16051, 23.16666667, 4.963333333, 389.16],
            [13.036, 144.3146667, 4847, 18.63333333, 18.36666667, 195.5966667],
            [23.44833333, 124.745, 7388, 16.9, 22.05333333, 235.5166667],
            [27.29866667, 106.2463333, 16051, 23.16666667, 4.963333333, 389.16],
            [16.36666667, 106.1196667, 1132, 27.23333333, 3.816666667, 519.9066667],
            [15.58633333, 144.378, 4857.666667, 19.56666667, 26.66666667, 241.21],
            [19.55066667, 128.865, 17387.33333, 35.83333333, 31.02, 456.8366667],
            [11.40066667, 110.0233333, 5435.666667, 19.76666667, 57.49666667, 203.06],
            [16.36666667, 106.1196667, 1132, 27.23333333, 3.816666667, 519.9066667],
            [11.09133333, 157.6246667, 2833, 26.13333333, 19.37, 181.11],
            [16.36666667, 106.1196667, 1132, 27.23333333, 3.816666667, 519.9066667],
            [13.971, 141.1563333, 5813, 21, 28.95666667, 227.61],
            [15.44266667, 249.443, 3109.666667, 21.23333333, 13.12, 212.54],
            [23.44833333, 124.745, 7388, 16.9, 22.05333333, 235.5166667],
            [15.81566667, 136.939, 1528.333333, 22.16666667, 19.69, 225.14],
            [19.55066667, 128.865, 17387.33333, 35.83333333, 31.02, 456.8366667],
            [13.693, 156.29, 2190, 15.36666667, 17.76666667, 416.7166667]
        ])
        expected_eigen_vectors = array([
            [0.429833643, -0.542626346, 0.37302348, 0.056694005, 0.024412442, -0.614689346],
            [-0.409761751, -0.052312821, 0.035590093, 0.848510303, 0.302681929, -0.12847543],
            [0.358127487, 0.056116642, 0.655436435, 0.211626614, 0.088065397, 0.621656953],
            [0.414472986, 0.628987545, -0.062158334, 0.362957533, -0.462622349, -0.288037402],
            [-0.289579131, 0.526819505, 0.515794378, -0.316511977, 0.369000286, -0.36908028],
            [-0.513263207, -0.162809143, 0.400102998, -0.010593811, -0.741519736, -0.002811768]
        ])

        generate_bootstraped_dataset_mock.side_effect = [stub_bootstraped]
        _, empiric_eigen_vectors, _ = apply_pca(DATA)

        # Act
        result_sample, result_eigen_vectors = bootstrap_and_apply_pca(DATA, empiric_eigen_vectors)

        # Assert
        self.assertTrue(allclose(stub_bootstraped, result_sample))
        self.assertTrue(allclose(expected_eigen_vectors, result_eigen_vectors))

    def test_generate_bootstraped_pcas_on_indicators(self):
        """
        Tests the method `generate_bootstraped_pcas_on_indicators` under the normal scenario.
        """

        # Arrange
        _, empiric_eigen_vectors, _ = apply_pca(DATA)

        # Act
        bootstraped_results, pcas_results = generate_bootstraped_pcas_on_indicators(
            DATA,
            empiric_eigen_vectors
        )

        # Assert
        self.assertEqual(len(bootstraped_results), NUMBER_OF_SAMPLES)
        self.assertEqual(len(pcas_results), NUMBER_OF_SAMPLES)

    def test_jacknife_and_apply_pca(self):
        """
        Tests the method `jacknifed_and_apply_pca` under the normal scenario.
        """

        # Act
        jacknifed_data, jacknifed_pca = jacknife_and_apply_pca(DATA)

        # Assert
        self.assertEqual(len(jacknifed_data), len(jacknifed_pca))
        self.assertTrue(allclose(abs(array(self.jacknifed_pcas)), abs(array(jacknifed_pca))))

    def test_confidence_intervals_from_indexes(self):
        """
        Tests the method `confidence_intervals_from_indexes` under the normal scenario.
        """

        # Arrange
        indexes = array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 3, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 2, 3, 0, 0, 0],
        ])

        # Act
        result = confidence_intervals_from_indexes(indexes, self.bootstraped_pcas)

        # Assert
        self.assertTrue(allclose(LOWER_BOUNDS, result))

    def test_produce_confidence_intervals(self):
        """
        Tests the method `produce_confidence_intervals` under the normal scenario.
        """

        # Act
        lower_bounds_results, upper_bounds_results = produce_confidence_intervals(
            self.bootstraped_pcas,
            self.jacknifed_pcas,
            EIGEN_VECTORS,
            0.05
        )

        # Assert
        self.assertTrue(allclose(LOWER_BOUNDS, lower_bounds_results))
        self.assertTrue(allclose(UPPER_BOUNDS, upper_bounds_results))

    def test_bootstraped_indicators_to_dataframe(self):
        """
        Tests the method `bootstraped_indicators_to_dataframe` under the normal scenario.
        """
        # Arrange
        _, empiric_eigen_vectors, _ = apply_pca(DATA)
        samples, _ = generate_bootstraped_pcas_on_indicators(DATA, empiric_eigen_vectors)

        # Act
        samples_dataframe = bootstraped_indicators_to_dataframe(samples, CODES)

        # Assert
        for row in samples_dataframe.itertuples():
            i = row.Index
            bootstrap_number = i // 18
            bootstrap_row = i % 18
            self.assertEqual(bootstrap_number + 1, row._1) # pylint: disable=protected-access
            self.assertEqual(samples[bootstrap_number][bootstrap_row][0], row.cei_pc020)
            self.assertEqual(samples[bootstrap_number][bootstrap_row][1], row.cei_pc030)
            self.assertEqual(samples[bootstrap_number][bootstrap_row][2], row.cei_pc034)
            self.assertEqual(samples[bootstrap_number][bootstrap_row][3], row.sdg_01_10)
            self.assertEqual(samples[bootstrap_number][bootstrap_row][4], row.sdg_06_40)
            self.assertEqual(samples[bootstrap_number][bootstrap_row][5], row.sdg_03_42)

    def test_jacknifed_indicators_to_dataframe(self):
        """
        Tests the method `jacknifed_indicators_to_dataframe` under the normal scenario.
        """
        # Arrange
        jacknifed = jacknife(DATA)

        # Act
        samples_dataframe = jacknifed_indicators_to_dataframe(jacknifed, CODES)

        # Assert
        for row in samples_dataframe.itertuples():
            i = row.Index
            jacknife_number = i // 17
            jacknife_row = i % 17
            self.assertEqual(jacknife_number + 1, row._1) # pylint: disable=protected-access
            self.assertEqual(jacknifed[jacknife_number][jacknife_row][0], row.cei_pc020)
            self.assertEqual(jacknifed[jacknife_number][jacknife_row][1], row.cei_pc030)
            self.assertEqual(jacknifed[jacknife_number][jacknife_row][2], row.cei_pc034)
            self.assertEqual(jacknifed[jacknife_number][jacknife_row][3], row.sdg_01_10)
            self.assertEqual(jacknifed[jacknife_number][jacknife_row][4], row.sdg_06_40)
            self.assertEqual(jacknifed[jacknife_number][jacknife_row][5], row.sdg_03_42)

    def test_confidence_interval_to_dataframe(self):
        """
        Tests the method `confidence_interval_to_dataframe` under the normal scenario.
        """

        # act
        intervals_dataframe = confidence_interval_to_dataframe(LOWER_BOUNDS, UPPER_BOUNDS, CODES)

        # Assert
        for i, row in enumerate(intervals_dataframe.itertuples()):
            if i < 6:
                self.assertEqual(CODES[i], row.indicator)
                self.assertEqual('lb', row._2) # pylint: disable=protected-access
                self.assertEqual(LOWER_BOUNDS[i][0], row._3) # pylint: disable=protected-access
                self.assertEqual(LOWER_BOUNDS[i][1], row._4) # pylint: disable=protected-access
                self.assertEqual(LOWER_BOUNDS[i][2], row._5) # pylint: disable=protected-access
                self.assertEqual(LOWER_BOUNDS[i][3], row._6) # pylint: disable=protected-access
                self.assertEqual(LOWER_BOUNDS[i][4], row._7) # pylint: disable=protected-access
                self.assertEqual(LOWER_BOUNDS[i][5], row._8) # pylint: disable=protected-access
            else:
                self.assertEqual(CODES[i - 6], row.indicator)
                self.assertEqual('ub', row._2) # pylint: disable=protected-access
                self.assertEqual(UPPER_BOUNDS[i - 6][0], row._3) # pylint: disable=protected-access
                self.assertEqual(UPPER_BOUNDS[i - 6][1], row._4) # pylint: disable=protected-access
                self.assertEqual(UPPER_BOUNDS[i - 6][2], row._5) # pylint: disable=protected-access
                self.assertEqual(UPPER_BOUNDS[i - 6][3], row._6) # pylint: disable=protected-access
                self.assertEqual(UPPER_BOUNDS[i - 6][4], row._7) # pylint: disable=protected-access
                self.assertEqual(UPPER_BOUNDS[i - 6][5], row._8) # pylint: disable=protected-access
