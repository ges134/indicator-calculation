"""
This module provides automated tests for the `Merger` module.
"""

from unittest import TestCase
from unittest.mock import patch, Mock
from pyjstat.pyjstat import Dataset
from pandas import DataFrame

from merger import merge_datasets, dataset_can_be_merged
from data import load_file

class MergerTests(TestCase):
    """
    This class provides automated tests for the module's methods.
    """

    @patch('merger.load_dataset')
    def test_merge_datasets(self, load_dataset_mock: Mock):
        """
        This method tests the `merge_datasets` function under the nominal scenario.

        For the moment, this method tests if the test data can be merged.
        """

        # Arrange
        cei_pc020_stub = Dataset.read(load_file('tests/cei_pc020.json'))
        cei_pc030_stub = Dataset.read(load_file('tests/cei_pc030.json'))
        cei_pc034_stub = Dataset.read(load_file('tests/cei_pc034.json'))

        cei_pc020_dataframe = cei_pc020_stub.write('dataframe')
        cei_pc030_dataframe = cei_pc030_stub.write('dataframe')
        cei_pc034_dataframe = cei_pc034_stub.write('dataframe')

        load_dataset_mock.side_effect =  [
            cei_pc020_dataframe,
            cei_pc030_dataframe,
            cei_pc034_dataframe
        ]

        # Act
        results = merge_datasets(['cei_pc020', 'cei_pc030', 'cei_pc034'])

        # assert
        load_dataset_mock.assert_any_call('cei_pc020')
        load_dataset_mock.assert_any_call('cei_pc030')
        load_dataset_mock.assert_any_call('cei_pc034')

        self.assertEqual(3, len(results))
        self.assertTrue(results[0])
        self.assertTrue(results[1])
        self.assertTrue(results[2])

    def test_dataset_can_be_merged(self):
        """
        This method tests the `dataset_can_be_merged` under the nominal scenario.
        """

        # Arrange
        data = {
            'Time frequency': ['Annual'],
            'Unit of measure': ['kg'], 
            'Geopolitical entity (reporting)': ['Tests'],
            'Time': [2025]
        }
        dataframe = DataFrame(data)

        # Act
        result = dataset_can_be_merged(dataframe)

        # Assert
        self.assertTrue(result)

    def test_dataset_can_be_merged_no_time_frequency(self):
        """
        This method thests the `dataset_can_be_merged` in a dataset with no time frequency.
        """

        # Arrange
        data = {
            'Unit of measure': ['kg'], 
            'Geopolitical entity (reporting)': ['Tests'],
        }
        dataframe = DataFrame(data)

        # Act
        result = dataset_can_be_merged(dataframe)

        # Assert
        self.assertFalse(result)

    def test_dataset_can_be_merged_no_unit_of_measure(self):
        """
        This method thests the `dataset_can_be_merged` in a dataset with no unit of measure.
        """

        # Arrange
        data = {
            'Time frequency': ['Annual'],
            'Geopolitical entity (reporting)': ['Tests'],
            'Time': [2025]
        }
        dataframe = DataFrame(data)

        # Act
        result = dataset_can_be_merged(dataframe)

        # Assert
        self.assertFalse(result)
