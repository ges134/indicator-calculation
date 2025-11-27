"""
This module offers automated tests for the `Merger` module.
"""

from unittest import TestCase
from unittest.mock import patch, Mock
from pyjstat.pyjstat import Dataset
from pandas import DataFrame
from math import isnan
from typing import List

from merger import (
    merge_datasets,
    dataset_can_be_merged,
    convert_dataset_to_dataframe,
    monitor_dataset,
    get_observations_with_complete_years,
    get_years_to_compute
)
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
    This class offers automated tests for the module's methods.
    """

    dataframes: List[DataFrame]

    def setUp(self):
        codes = ['cei_pc020', 'cei_pc030', 'cei_pc034', 'sdg_01_10', 'sdg_03_42']
        dataframes = []
        for code in codes:
            stub = Dataset.read(load_file(f'tests/{code}.json'))
            dataframes.append(stub.write('dataframe'))

        self.dataframes = dataframes

    @patch('merger.load_dataset')
    def test_merge_datasets(self, load_dataset_mock: Mock):
        """
        This method tests the `merge_datasets` function under the nominal scenario.
        """

        # Arrange
        load_dataset_mock.side_effect = self.dataframes

        # Act
        results = merge_datasets(CONFIG)

        # assert
        load_dataset_mock.assert_any_call('cei_pc020')
        load_dataset_mock.assert_any_call('cei_pc030')
        load_dataset_mock.assert_any_call('cei_pc034')
        load_dataset_mock.assert_any_call('sdg_01_10')
        load_dataset_mock.assert_any_call('sdg_03_42')

        verified_row_keys = []

        for row in self.dataframes[0].itertuples():
            geopolitical_column_location = self.dataframes[0].columns.get_loc(
                'Geopolitical entity (reporting)'
            )
            row_key = f'{row[geopolitical_column_location+1]};{row.Time}'
            if not isnan(results[row_key]['values']['EMA']):
                self.assertEqual(results[row_key]['values']['EMA'], row.value)

            if row_key not in verified_row_keys:
                self.assertTrue(1 <= len(results[row_key]['values']) <= 5)
                verified_row_keys.append(row_key)

        for row in self.dataframes[1][
            self.dataframes[1]['Unit of measure'] == UNIT_OF_MEASURE_LABEL
        ].itertuples():
            geopolitical_column_location = self.dataframes[1].columns.get_loc(
                'Geopolitical entity (reporting)'
            )
            row_key = f'{row[geopolitical_column_location+1]};{row.Time}'
            if not isnan(results[row_key]['values']['PDR']):
                self.assertEqual(results[row_key]['values']['PDR'], row.value)

            if row_key not in verified_row_keys:
                self.assertTrue(1 <= len(results[row_key]['values']) <= 5)
                verified_row_keys.append(row_key)

        for row in self.dataframes[2].itertuples():
            geopolitical_column_location = self.dataframes[2].columns.get_loc(
                'Geopolitical entity (reporting)'
            )
            row_key = f'{row[geopolitical_column_location+1]};{row.Time}'
            if not isnan(results[row_key]['values']['GMR']):
                self.assertEqual(results[row_key]['values']['GMR'], row.value)

            if row_key not in verified_row_keys:
                self.assertTrue(1 <= len(results[row_key]['values']) <= 5)
                verified_row_keys.append(row_key)

        for row in self.dataframes[3][
            (self.dataframes[3]['Age class'] == 'Total') &
            (self.dataframes[3]['Unit of measure'] == "Thousand persons")
        ].itertuples():
            geopolitical_column_location = self.dataframes[3].columns.get_loc(
                'Geopolitical entity (reporting)'
            )
            row_key = f'{row[geopolitical_column_location+1]};{row.Time}'
            if not isnan(results[row_key]['values']['TRP']):
                self.assertEqual(results[row_key]['values']['TRP'], row.value)

            if row_key not in verified_row_keys:
                self.assertTrue(1 <= len(results[row_key]['values']) <= 5)
                verified_row_keys.append(row_key)

        for row in self.dataframes[4][
            self.dataframes[4]['Type of mortality'] == 'Preventable mortality'
        ].itertuples():
            geopolitical_column_location = self.dataframes[4].columns.get_loc(
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
    def test_convert_dataset_to_dataframe(self, load_dataset_mock: Mock):
        """
        This method tests the `convert_dataset_to_dataframe` under the nominal scenario.
        """

        # Arrange
        load_dataset_mock.side_effect = self.dataframes

        merged = merge_datasets(CONFIG)

        # Act
        results = convert_dataset_to_dataframe(merged, CONFIG)

        # Assert
        verified_row_keys = []

        for row in self.dataframes[0].itertuples():
            geopolitical_column_location = self.dataframes[0].columns.get_loc(
                'Geopolitical entity (reporting)'
            )
            country = row[geopolitical_column_location+1]
            year = int(row.Time)
            row_key = f'{country};{year}'
            result_row = results[(results.Country == country) & (results.Year == year)]
            result_value = result_row.EMA.iat[0]

            if not isnan(result_value):
                self.assertEqual(result_value, row.value)

            if row_key not in verified_row_keys:
                verified_row_keys.append(row_key)

        for row in self.dataframes[1][
            self.dataframes[1]['Unit of measure'] == UNIT_OF_MEASURE_LABEL
        ].itertuples():
            geopolitical_column_location = self.dataframes[1].columns.get_loc(
                'Geopolitical entity (reporting)'
            )
            country = row[geopolitical_column_location+1]
            year = int(row.Time)
            row_key = f'{country};{year}'
            result_row = results[(results.Country == country) & (results.Year == year)]
            result_value = result_row.PDR.iat[0]

            if not isnan(result_value):
                self.assertEqual(result_value, row.value)

            if row_key not in verified_row_keys:
                verified_row_keys.append(row_key)

        for row in self.dataframes[2].itertuples():
            geopolitical_column_location = self.dataframes[2].columns.get_loc(
                'Geopolitical entity (reporting)'
            )
            country = row[geopolitical_column_location+1]
            year = int(row.Time)
            row_key = f'{country};{year}'
            result_row = results[(results.Country == country) & (results.Year == year)]
            result_value = result_row.GMR.iat[0]

            if not isnan(result_value):
                self.assertEqual(result_value, row.value)

            if row_key not in verified_row_keys:
                verified_row_keys.append(row_key)

        for row in self.dataframes[3][
            (self.dataframes[3]['Age class'] == 'Total') &
            (self.dataframes[3]['Unit of measure'] == "Thousand persons")
        ].itertuples():
            geopolitical_column_location = self.dataframes[3].columns.get_loc(
                'Geopolitical entity (reporting)'
            )
            country = row[geopolitical_column_location+1]
            year = int(row.Time)
            row_key = f'{country};{year}'
            result_row = results[(results.Country == country) & (results.Year == year)]
            result_value = result_row.TRP.iat[0]

            if not isnan(result_value):
                self.assertEqual(result_value, row.value)

            if row_key not in verified_row_keys:
                verified_row_keys.append(row_key)

        for row in self.dataframes[4][
            self.dataframes[4]['Type of mortality'] == 'Preventable mortality'
        ].itertuples():
            geopolitical_column_location = self.dataframes[4].columns.get_loc(
                'Geopolitical entity (reporting)'
            )
            country = row[geopolitical_column_location+1]
            year = int(row.Time)
            row_key = f'{country};{year}'
            result_row = results[(results.Country == country) & (results.Year == year)]
            result_value = result_row.PMS.iat[0]

            if not isnan(result_value):
                self.assertEqual(result_value, row.value)

            if row_key not in verified_row_keys:
                verified_row_keys.append(row_key)

        self.assertEqual(len(results), len(verified_row_keys))
        self.assertEqual(results.shape[1], 7)

    @patch('merger.load_dataset')
    def test_monitor_dataset(self, load_dataset_mock: Mock):
        """
        Tests the metho `monitor_dataset` under the nominal scenario.
        """

        # Arrange
        codes = ['cei_pc020', 'cei_pc030', 'cei_pc034', 'sdg_01_10', 'sdg_03_42']
        dataframes = []
        for code in codes:
            stub = Dataset.read(load_file(f'tests/{code}.json'))
            dataframes.append(stub.write('dataframe'))

        load_dataset_mock.side_effect = dataframes
        results = merge_datasets(CONFIG)
        reference = convert_dataset_to_dataframe(results, CONFIG)
        new_dataset = reference.copy()
        new_dataset.loc[0, 'PMS'] = 10.0
        new_dataset.loc[72, 'PDR'] = 20.0

        # Act
        results = monitor_dataset(reference, new_dataset)

        # Assert
        self.assertEqual(2, len(results.index))
        self.assertEqual(4, len(results.columns))
        self.assertTrue(isnan(results.loc[0, 'PDR'].reference))
        self.assertTrue(isnan(results.loc[0, 'PDR'].new))
        self.assertTrue(isnan(results.loc[0, 'PMS'].reference))
        self.assertEqual(10.0, results.loc[0, 'PMS'].new)
        self.assertEqual(2.1291, results.loc[72, 'PDR'].reference)
        self.assertEqual(20.0, results.loc[72, 'PDR'].new)
        self.assertTrue(isnan(results.loc[72, 'PMS'].reference))
        self.assertTrue(isnan(results.loc[72, 'PMS'].new))

    @patch('merger.load_dataset')
    def test_get_observations_with_complete_years(self, load_dataset_mock: Mock):
        """
        Tests the method `get_observations_with_complete_years` under the typical scenario
        """

        # Arrange
        load_dataset_mock.side_effect = self.dataframes
        merged = merge_datasets(CONFIG)
        merged_dataframe = convert_dataset_to_dataframe(merged, CONFIG)
        expected_years = [2016, 2018, 2020]

        # Act
        complete_observations = get_observations_with_complete_years(merged_dataframe)

        # Assert
        years = complete_observations.Year.unique()
        self.assertEqual(
            0,
            len(complete_observations[complete_observations.Country.str.contains('European Union')])
        )
        self.assertFalse(complete_observations.isnull().any().any())
        self.assertEqual(expected_years, years.tolist())
        for year in years:
            self.assertEqual(
                28,
                len(complete_observations[complete_observations.Year == year])
            )

    @patch('merger.load_dataset')
    def test_get_years_to_compute(self, load_dataset_mock: Mock):
        """
        Tests the method `get_observations_with_complete_years` under the typical scenario
        """

        # Arrange
        load_dataset_mock.side_effect = self.dataframes
        merged = merge_datasets(CONFIG)
        merged_dataframe = convert_dataset_to_dataframe(merged, CONFIG)
        expected_years = [2016, 2018, 2020]
        complete_observations = get_observations_with_complete_years(merged_dataframe)

        # Act
        years = get_years_to_compute(complete_observations)

        # Assert
        self.assertEqual(expected_years, years)
