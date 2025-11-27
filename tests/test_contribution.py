"""
This module offers automated tests for the `Contribution` module.
"""

from unittest import TestCase
from matplotlib.testing.compare import compare_images

from contribution import make_loading_plot, make_loading_plot_with_confidence_intervals
from stats import apply_pca
from tests.constants import DATA, EIGEN_VECTORS, LOWER_BOUNDS, UPPER_BOUNDS

CODES = ['EMA', 'PDR', 'GMR', 'TRP', 'NDE', 'PMS']

class TestContribution(TestCase):
    """
    This class offers automated tests for the module's methods.
    """

    def test_make_loading_plot(self):
        """
        Tests the method `make_loading_plot` under the typical scenario.
        """

        # Arrange
        _, eigen_vectors, _ = apply_pca(DATA)

        # Act
        make_loading_plot(eigen_vectors, CODES, './data/tests/contribution_actual.png')

        # Assert
        results = compare_images(
            'data/tests/contribution_baseline.png',
            'data/tests/contribution_actual.png',
            0.001
        )
        self.assertIsNone(results)

    def test_make_loading_plot_with_confidence_intervals(self):
        """
        Tests the method `make_loading_plot_with_confidence_intervals` under the typical scenario.
        """

        # Act
        make_loading_plot_with_confidence_intervals(
            EIGEN_VECTORS,
            CODES,
            './data/tests/contribution_intervals_actual.png',
            LOWER_BOUNDS,
            UPPER_BOUNDS
        )

        # Assert
        results = compare_images(
            'data/tests/contribution_intervals_baseline.png',
            'data/tests/contribution_intervals_actual.png',
            0.001
        )
        self.assertIsNone(results)
