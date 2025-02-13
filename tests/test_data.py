"""
This module provides automated tests for the `Data` module.
"""

from unittest import TestCase
from typing import Dict

from data import transform_units_from_file

class DataTests(TestCase):
    """
    This class provides automated tests for the module's methods.
    """

    def test_transform_units_from_file(self) -> Dict:
        """
        This method tests the `transform_units_from_file` function under the nominal scenario.
        """

        # Arrange
        unit_input = 'cei_pc030 UNIT Euro per kilogram, chain linked volumes (2015)'

        # Act
        results = transform_units_from_file(unit_input)

        # Assert
        self.assertEqual(len(results), 1)
        self.assertEqual('Euro per kilogram, chain linked volumes (2015)', results['cei_pc030'])
