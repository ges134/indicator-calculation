"""
This module provides automated tests for the `Merger` module.
"""

from unittest import TestCase
from unittest.mock import patch, Mock
from pyjstat.pyjstat import Dataset
from pandas import DataFrame
from math import isnan

from merger import merge_datasets, dataset_can_be_merged, merged_dataset_to_csv
from data import load_file

UNIT_OF_MEASURE_LABEL = 'Euro per kilogram, chain linked volumes (2015)'
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
    },
    {
        'id': 'TRP',
        'code': 'sdg_01_10',
        'Age class': 'Total',
        'Unit of measure': 'Thousand persons'
    },
    {
        'id': 'PMS',
        'code': 'sdg_03_42',
        'Type of mortality': 'Preventable mortality'
    }
]

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
        codes = ['cei_pc020', 'cei_pc030', 'cei_pc034', 'sdg_01_10', 'sdg_03_42']
        dataframes = []
        for code in codes:
            stub = Dataset.read(load_file(f'tests/{code}.json'))
            dataframes.append(stub.write('dataframe'))

        load_dataset_mock.side_effect = dataframes

        # Act
        results = merge_datasets(CONFIG)

        # assert
        load_dataset_mock.assert_any_call('cei_pc020')
        load_dataset_mock.assert_any_call('cei_pc030')
        load_dataset_mock.assert_any_call('cei_pc034')
        load_dataset_mock.assert_any_call('sdg_01_10')
        load_dataset_mock.assert_any_call('sdg_03_42')

        verified_row_keys = []

        for row in dataframes[0].itertuples():
            geopolitical_column_location = dataframes[0].columns.get_loc(
                'Geopolitical entity (reporting)'
            )
            row_key = f'{row[geopolitical_column_location+1]};{row.Time}'
            if not isnan(results[row_key]['values']['EMA']):
                self.assertEqual(results[row_key]['values']['EMA'], row.value)

            if row_key not in verified_row_keys:
                self.assertTrue(1 <= len(results[row_key]['values']) <= 5)
                verified_row_keys.append(row_key)

        for row in dataframes[1][
            dataframes[1]['Unit of measure'] == UNIT_OF_MEASURE_LABEL
        ].itertuples():
            geopolitical_column_location = dataframes[1].columns.get_loc(
                'Geopolitical entity (reporting)'
            )
            row_key = f'{row[geopolitical_column_location+1]};{row.Time}'
            if not isnan(results[row_key]['values']['PDR']):
                self.assertEqual(results[row_key]['values']['PDR'], row.value)

            if row_key not in verified_row_keys:
                self.assertTrue(1 <= len(results[row_key]['values']) <= 5)
                verified_row_keys.append(row_key)

        for row in dataframes[2].itertuples():
            geopolitical_column_location = dataframes[2].columns.get_loc(
                'Geopolitical entity (reporting)'
            )
            row_key = f'{row[geopolitical_column_location+1]};{row.Time}'
            if not isnan(results[row_key]['values']['GMR']):
                self.assertEqual(results[row_key]['values']['GMR'], row.value)

            if row_key not in verified_row_keys:
                self.assertTrue(1 <= len(results[row_key]['values']) <= 5)
                verified_row_keys.append(row_key)

        for row in dataframes[3][
            (dataframes[3]['Age class'] == 'Total') &
            (dataframes[3]['Unit of measure'] == "Thousand persons")
        ].itertuples():
            geopolitical_column_location = dataframes[3].columns.get_loc(
                'Geopolitical entity (reporting)'
            )
            row_key = f'{row[geopolitical_column_location+1]};{row.Time}'
            if not isnan(results[row_key]['values']['TRP']):
                self.assertEqual(results[row_key]['values']['TRP'], row.value)

            if row_key not in verified_row_keys:
                self.assertTrue(1 <= len(results[row_key]['values']) <= 5)
                verified_row_keys.append(row_key)

        for row in dataframes[4][
            dataframes[4]['Type of mortality'] == 'Preventable mortality'
        ].itertuples():
            geopolitical_column_location = dataframes[4].columns.get_loc(
                'Geopolitical entity (reporting)'
            )
            row_key = f'{row[geopolitical_column_location+1]};{row.Time}'
            if not isnan(results[row_key]['values']['PMS']):
                self.assertEqual(results[row_key]['values']['PMS'], row.value)

            if row_key not in verified_row_keys:
                self.assertTrue(1 <= len(results[row_key]['values']) <= 5)
                verified_row_keys.append(row_key)

        self.assertEqual(len(results), len(verified_row_keys))

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

    def test_dataset_can_be_merged_no_time(self):
        """
        This method thests the `dataset_can_be_merged` in a dataset with no time.
        """

        # Arrange
        data = {
            'Unit of measure': ['kg'], 
            'Geopolitical entity (reporting)': ['Tests'],
            'Time frequency': ['Annual'],
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
        codes = ['cei_pc020', 'cei_pc030', 'cei_pc034', 'sdg_01_10', 'sdg_03_42']
        dataframes = []
        for code in codes:
            stub = Dataset.read(load_file(f'tests/{code}.json'))
            dataframes.append(stub.write('dataframe'))

        load_dataset_mock.side_effect = dataframes

        merged = merge_datasets(CONFIG)

        # Act
        results = merged_dataset_to_csv(merged, CONFIG)

        # Assert
        verified_row_keys = []

        for row in dataframes[0].itertuples():
            geopolitical_column_location = dataframes[0].columns.get_loc(
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

        for row in dataframes[1][
            dataframes[1]['Unit of measure'] == UNIT_OF_MEASURE_LABEL
        ].itertuples():
            geopolitical_column_location = dataframes[1].columns.get_loc(
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

        for row in dataframes[2].itertuples():
            geopolitical_column_location = dataframes[2].columns.get_loc(
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

        for row in dataframes[3][
            (dataframes[3]['Age class'] == 'Total') &
            (dataframes[3]['Unit of measure'] == "Thousand persons")
        ].itertuples():
            geopolitical_column_location = dataframes[3].columns.get_loc(
                'Geopolitical entity (reporting)'
            )
            country = row[geopolitical_column_location+1]
            year = row.Time
            row_key = f'{country};{year}'
            result_row = [row for row in results if country == row[0] and year == row[1]][0]

            if result_row[5] != '':
                self.assertEqual(result_row[5], row.value)

            if row_key not in verified_row_keys:
                verified_row_keys.append(row_key)

        for row in dataframes[4][
            dataframes[4]['Type of mortality'] == 'Preventable mortality'
        ].itertuples():
            geopolitical_column_location = dataframes[4].columns.get_loc(
                'Geopolitical entity (reporting)'
            )
            country = row[geopolitical_column_location+1]
            year = row.Time
            row_key = f'{country};{year}'
            result_row = [row for row in results if country == row[0] and year == row[1]][0]

            if result_row[6] != '':
                self.assertEqual(result_row[6], row.value)

            if row_key not in verified_row_keys:
                verified_row_keys.append(row_key)

        self.assertEqual(len(results)-1, len(verified_row_keys))
        self.assertTrue(all(len(row) == 7 for row in results))
