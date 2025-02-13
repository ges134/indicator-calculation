"""
This module provides automated tests for the `Merger` module.
"""

from unittest import TestCase
from unittest.mock import patch, Mock
from pyjstat.pyjstat import Dataset
from pandas import DataFrame
from math import isnan

from merger import merge_datasets, dataset_can_be_merged, merged_dataset_to_csv, units_to_csv
from data import load_file

UNIT_OF_MEASURE_LABEL = 'Euro per kilogram, chain linked volumes (2015)'

class MergerTests(TestCase):
    """
    This class provides automated tests for the module's methods.
    """

    @patch('merger.load_dataset')
    def test_merge_datasets(self, load_dataset_mock: Mock):
        """
        This method tests the `merge_datasets` function under the nominal scenario.
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

        units_input = {
            'cei_pc030': 'Euro per kilogram, chain linked volumes (2015)'
        }

        # Act
        results, units = merge_datasets(['cei_pc020', 'cei_pc030', 'cei_pc034'], units_input)

        # assert
        load_dataset_mock.assert_any_call('cei_pc020')
        load_dataset_mock.assert_any_call('cei_pc030')
        load_dataset_mock.assert_any_call('cei_pc034')

        verified_row_keys = []

        for row in cei_pc020_dataframe.itertuples():
            geopolitical_column_location = cei_pc020_dataframe.columns.get_loc(
                'Geopolitical entity (reporting)'
            )
            row_key = f'{row[geopolitical_column_location+1]};{row.Time}'
            if not isnan(results[row_key]['values']['cei_pc020']):
                self.assertEqual(results[row_key]['values']['cei_pc020'], row.value)

            if row_key not in verified_row_keys:
                self.assertTrue(1 <= len(results[row_key]['values']) <= 3)
                verified_row_keys.append(row_key)

        for row in cei_pc030_dataframe[
            cei_pc030_dataframe['Unit of measure'] == UNIT_OF_MEASURE_LABEL
        ].itertuples():
            geopolitical_column_location = cei_pc030_dataframe.columns.get_loc(
                'Geopolitical entity (reporting)'
            )
            row_key = f'{row[geopolitical_column_location+1]};{row.Time}'
            if not isnan(results[row_key]['values']['cei_pc030']):
                self.assertEqual(results[row_key]['values']['cei_pc030'], row.value)

            if row_key not in verified_row_keys:
                self.assertTrue(1 <= len(results[row_key]['values']) <= 3)
                verified_row_keys.append(row_key)

        for row in cei_pc034_dataframe.itertuples():
            geopolitical_column_location = cei_pc034_dataframe.columns.get_loc(
                'Geopolitical entity (reporting)'
            )
            row_key = f'{row[geopolitical_column_location+1]};{row.Time}'
            if not isnan(results[row_key]['values']['cei_pc034']):
                self.assertEqual(results[row_key]['values']['cei_pc034'], row.value)

            if row_key not in verified_row_keys:
                self.assertTrue(1 <= len(results[row_key]['values']) <= 3)
                verified_row_keys.append(row_key)

        self.assertEqual(len(results), len(verified_row_keys))
        self.assertEqual(3, len(units))
        self.assertEqual('Tonnes per capita', units['cei_pc020'])
        self.assertEqual('Euro per kilogram, chain linked volumes (2015)', units['cei_pc030'])
        self.assertEqual('Kilograms per capita', units['cei_pc034'])


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
        This method tests the `dataset_can_be_merged` in a dataset with no unit of measure.
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

    @patch('merger.load_dataset')
    def test_merged_dataset_to_csv(self, load_dataset_mock: Mock):
        """
        This method tests the `merged_dataset_to_csv` under the nominal scenario.
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
        codes = ['cei_pc020', 'cei_pc030', 'cei_pc034']

        merged, _ = merge_datasets(codes, {})

        # Act
        results = merged_dataset_to_csv(merged, codes)

        # Assert
        verified_row_keys = []

        for row in cei_pc020_dataframe.itertuples():
            geopolitical_column_location = cei_pc020_dataframe.columns.get_loc(
                'Geopolitical entity (reporting)'
            )
            country = row[geopolitical_column_location+1]
            year = row.Time
            row_key = f'{country};{year}'
            result_row = [row for row in results if country == row[0] and year == row[1]][0]

            if result_row[2] != '':
                self.assertEqual(result_row[2], row.value)

            if row_key not in verified_row_keys:
                verified_row_keys.append(row_key)

        for row in cei_pc030_dataframe[
            cei_pc030_dataframe['Unit of measure'] == UNIT_OF_MEASURE_LABEL
        ].itertuples():
            geopolitical_column_location = cei_pc030_dataframe.columns.get_loc(
                'Geopolitical entity (reporting)'
            )
            country = row[geopolitical_column_location+1]
            year = row.Time
            row_key = f'{country};{year}'
            result_row = [row for row in results if country == row[0] and year == row[1]][0]

            if result_row[3] != '':
                self.assertEqual(result_row[3], row.value)

            if row_key not in verified_row_keys:
                verified_row_keys.append(row_key)

        for row in cei_pc034_dataframe.itertuples():
            geopolitical_column_location = cei_pc034_dataframe.columns.get_loc(
                'Geopolitical entity (reporting)'
            )
            country = row[geopolitical_column_location+1]
            year = row.Time
            row_key = f'{country};{year}'
            result_row = [row for row in results if country == row[0] and year == row[1]][0]

            if result_row[4] != '':
                self.assertEqual(result_row[4], row.value)

            if row_key not in verified_row_keys:
                verified_row_keys.append(row_key)

        self.assertEqual(len(results)-1, len(verified_row_keys))
        self.assertTrue(all(len(row) == 5 for row in results))

    @patch('merger.load_dataset')
    def test_units_to_csv(self, load_dataset_mock: Mock):
        """
        This method tests the `units_to_csv` under the nominal scenario.
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
        codes = ['cei_pc020', 'cei_pc030', 'cei_pc034']
        units_input = {
            'cei_pc030': 'Euro per kilogram, chain linked volumes (2015)'
        }

        _, units = merge_datasets(codes, units_input)

        # Act
        results = units_to_csv(units)

        # Assert
        self.assertEqual(4, len(results))
        self.assertTrue(all(len(row) == 2 for row in results))
        self.assertEqual('cei_pc030', results[1][0])
        self.assertEqual('cei_pc020', results[2][0])
        self.assertEqual('cei_pc034', results[3][0])
        self.assertEqual('Euro per kilogram, chain linked volumes (2015)', results[1][1])
        self.assertEqual('Tonnes per capita', results[2][1])
        self.assertEqual('Kilograms per capita', results[3][1])
