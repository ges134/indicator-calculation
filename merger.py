"""
This module ensures that all the indicators' datasets are merged into one.
"""

from typing import List, Dict
from tqdm import tqdm
from pandas import DataFrame
from math import isnan

from data import load_dataset

def merge_datasets(config: List[Dict]) -> Dict:
    """
    This function applies the dataset merging from the `codes` provided in the parameter. This is
    the central function of merging.

    Each dataset is computed entirely before moving on to the next one. The computation does the
    following elements:
    
    1. Load the dataset and convert it into a `DataFrame`
    2. Check if the dataset is of appropriate format. If not, the computing stops there for this
    dataset.
    3. With the appropriate format, it scans each row and updates the merged dataset. New keys are
    created if they don't exists.

    Args:
        config: The configuration file for this program execution. It should provide an identifier,
            an Eurostat data code and the necessary specifications to read multiple dimensions. For
            instance, if there is multiple units of measure for one dataset, the configuration 
            should specify which unit to choose.
    
    Returns:
        A dictionnary with the merged datasets. In this dictionnary, each entry has a key with the
        country and the year of reporting, separated by a semi-colomn (;). Inside, another
        dictionnary with the values is stored. The values can be accessed with the `value` key. In 
        this dictionnary, the indicator code is the key and the indicator value is in the value.
    """

    merged = {}

    for indicator in tqdm(
        config,
        'Computation of datasets',
        leave=False
    ):
        code = indicator['code']
        id = indicator['id']
        dataframe = load_dataset(code)
        if dataset_can_be_merged(dataframe):
            dimensions = [d for d in indicator.keys() if d not in ('id', 'code')]

            for dimension in tqdm(dimensions, f'Preparing indicator {id} for merge', leave=False):
                dataframe = dataframe[dataframe[dimension] == indicator[dimension]]

            for row in tqdm(dataframe.itertuples(), f'Merging indicator {code}', leave=False):
                geopolitical_column_location = dataframe.columns.get_loc(
                    'Geopolitical entity (reporting)'
                )
                row_key = f'{row[geopolitical_column_location+1]};{row.Time}'
                if row_key not in merged:
                    merged[row_key]= {
                        'values': {}
                    }

                merged[row_key]['values'][id] = row.value

    return merged

def dataset_can_be_merged(dataframe: DataFrame) -> bool:
    """
    This function checks if the data frame given in the `dataframe` parameter 
    can be merged. A data frame can be merged if there is a annual time frequency
    and if the geopolitical entity and time columns are on the dataset.

    Args:
        dataframe: The dataframe to test.
    
    Returns:
        True if the dataset can be merged. False otherwise.
    """
    if (
        'Time frequency' not in dataframe.columns
        or dataframe['Time frequency'].iloc[0] != 'Annual'
    ):
        return False
    if (
        'Geopolitical entity (reporting)' not in dataframe.columns
        or 'Time' not in dataframe.columns
    ):
        return False
    return True

def convert_dataset_to_dataframe(merged: Dict, config: List[Dict]) -> DataFrame:
    """
    Converts the merged dataset into a Dataframe.

    The merged dataset can be generated with the `merge_datasets` method of this module.

    Args:
        merged: The dictionnary containing the merged dataset.
        config: The configuration file of this program execution.

    Returns:
        A dataframe of the merged datasets. This list can be saved into a file with the
        `data.save_csv` method.
    """

    codes = [c['id'] for c in config]
    colums = ['Country', 'Year'] + [c['id'] for c in config]
    data = []

    for key, entry in tqdm(
        merged.items(),
        'Conversion of dataset into a CSV file',
        leave=False
    ):
        country, year = key.split(';')
        row = [country, int(year)] + [None] * len(config)
        for indicator, value in entry['values'].items():
            index = codes.index(indicator)
            row[index+2] = None if isnan(value) else value

        data.append(row)

    dataframe = DataFrame(data, columns=colums)
    dataframe.sort_values(['Country', 'Year'])

    return dataframe
