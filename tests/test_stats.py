"""
This module provides automated tests for the `Stats` module.
"""

from unittest import TestCase
from numpy import array, allclose

from stats import (
    generate_bootstraped_dataset,
    jacknife,
    apply_pca,
    correlation_matrix_between_pcas,
    test_for_normality
)
from tests.constants import DATA

class TestStats(TestCase):
    """
    This class provides automated tests for the module's methods.
    """

    def test_generate_bootstraped_dataset(self):
        """
        Tests the method `generate_bootstraped_dataset` under the normal scenario.
        """

        # Act
        generated = generate_bootstraped_dataset(DATA)

        # Assert
        self.assertEqual(len(DATA), len(generated))
        for row in generated.tolist():
            self.assertTrue(row in DATA.tolist())

    def test_jacknife(self):
        """
        Tests the method `jacknife` under the normal scenario
        """

        # Act
        jacknifed = jacknife(DATA)

        # Assert
        self.assertEqual(len(DATA), len(jacknifed))
        self.assertTrue(all(len(s) == len(DATA) - 1 for s in jacknifed))
        for i, sample in enumerate(jacknifed):
            self.assertFalse(DATA[i].tolist() in sample.tolist())

    def test_apply_pca(self):
        """
        Tests the method `apply_pca` under the nominal scenario.
        """
        # Arrange
        expected_eigenvalues = [2.3724, 1.4218, 1.1748,	0.5915, 0.3011, 0.1383]
        expected_eigenvectors = [
            [0.541, -0.350, 0.193, 0.136, 0.145, 0.713],
            [-0.436, -0.236, -0.344, 0.696, 0.377, 0.099],
            [0.571, -0.006, 0.214, 0.405, 0.268, -0.626],
            [0.091, 0.744, 0.053, 0.491, -0.356, 0.260],
            [-0.321, 0.302, 0.662, -0.105, 0.587, 0.113],
            [0.283, 0.421, -0.599, -0.283, 0.543, 0.098]
        ]
        expected_variance = [0.395, 0.237, 0.196, 0.099, 0.050, 0.023]

        # Act
        eigen_values, eigen_vectors, explained_variance = apply_pca(DATA)

        # Assert
        for i, value in enumerate(eigen_values):
            self.assertAlmostEqual(value, expected_eigenvalues[i], 4)

        for i, row in enumerate(eigen_vectors):
            for j, value in enumerate(row):
                self.assertAlmostEqual(value, expected_eigenvectors[i][j], 3)

        for i, value in enumerate(explained_variance):
            self.assertAlmostEqual(value, expected_variance[i], 3)

    def test_correlation_matrix_between_pcas(self):
        """
        Tests the `correlation_matrix_between_pcas` under the normal scenario.
        """

        # Arrange
        eigen_vectors_a = array([
            [0.541064083, 0.349956854, -0.193075386, -0.136123106, -0.145444777, -0.71261355],
            [-0.43625019, 0.236016078, 0.343606096, -0.695991371, -0.376546102, -0.098620878],
            [0.57091438, 0.006257168, -0.213755634, -0.40527501, -0.267562105, 0.626489293],
            [0.091402506, -0.743777939, -0.053399282, -0.491407174, 0.355602335, -0.26010468],
            [-0.320585679, -0.302434357, -0.661534328, 0.105253131, -0.586747744, -0.113046356],
            [0.282599727, -0.420850682, 0.598737108, 0.28330098, -0.542902785, -0.097637565]
        ])
        eigen_vectors_b = array([
            [0.429833643, 0.542626346, 0.37302348, -0.056694005, -0.024412442, 0.614689346],
            [-0.409761751, 0.052312821, 0.035590093, -0.848510303, -0.302681929, 0.12847543],
            [0.358127487, -0.056116642, 0.655436435, -0.211626614, -0.088065397, -0.621656953],
            [0.414472986, -0.628987545, -0.062158334, -0.362957533, 0.462622349, 0.288037402],
            [-0.289579131, -0.526819505, 0.515794378, 0.316511977, -0.369000286, 0.36908028],
            [0.513263207, -0.162809143, -0.400102998, -0.010593811, -0.741519736, -0.002811768]
        ])
        expected_results = array([
            [0.884218976, 0.440762439, 0.165484908, 0.266955321, 0.172976954, -0.293139145],
            [-0.273649285, 0.885273, 0.438679968, -0.26090324, -0.185327077, 0.033850286],
            [0.22789856, 0.197266058, -0.838267321, -0.533304794, -0.374878731, -0.189940511],
            [0.288638958, -0.068306765, -0.16118515, 0.858227147, -0.61534332, 0.187235445],
            [0.42758136, -0.154842312, -0.086589398, -0.341275792, 0.933899719, 0.161786845],
            [-0.063505007, -0.299039589, 0.303616527, -0.091159399, -0.150088965, -0.953884874]
        ])

        # Act
        correlation_matrix = correlation_matrix_between_pcas(eigen_vectors_a, eigen_vectors_b)

        # Assert
        self.assertTrue(allclose(expected_results, correlation_matrix))

    def test_test_for_normality(self):
        """
        Tests the `test_for_normality` under the normal scenario.
        """

        # Arrange
        expected_results = [False, True, False, True, True, False]

        # Act
        results = test_for_normality(DATA)

        # Assert
        self.assertEqual(expected_results[0], results[0])
        self.assertEqual(expected_results[1], results[1])
        self.assertEqual(expected_results[2], results[2])
        self.assertEqual(expected_results[3], results[3])
        self.assertEqual(expected_results[4], results[4])
        self.assertEqual(expected_results[5], results[5])
