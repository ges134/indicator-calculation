"""
This module ensures that all indicator datasets are merged into a single dataset.
"""

from typing import List, Dict
from tqdm import tqdm
from pandas import DataFrame
from math import isnan

from data import load_dataset

def merge_datasets(config: List[Dict]) -> Dict:
    """
    This function applies the dataset merging with the specified configuration.

    Each dataset is computed entirely before moving on to the next one. The computation does the
    following:
    
    1. Load the dataset and convert it into a `DataFrame`
    2. Check if the dataset is of the appropriate format. If not, the computing stops there for this
    dataset.
    3. ith the appropriate format, it scans each row and updates the merged dataset. New keys are
    created if they do not exist.

    Args:
        - config: The configuration file for this program execution. It should provide an
            identifier, an Eurostat data code and the necessary specifications to read multiple
            dimensions. For instance, if there are multiple units of measure for a dataset, the
            configuration should specify which unit to use.
    
    Returns: A dictionary with the merged datasets. In this dictionary, each entry has a key with
        the country and the year of reporting, separated by a semicolon (;). Inside, another
        dictionary with the values is stored. The values can be accessed with the `value` key. In
        this dictionary, the indicator code is the key, and the indicator value is in the value.
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
            dimensions = [d for d in indicator.keys() if d not in ('id', 'code', 'social', 'environmental', 'economic')]

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
    This function checks whether the data frame provided in the `dataframe` parameter can be merged.
    A data frame can be merged if the time frequency is annual and the dataset includes the
    geopolitical entity and time columns.

    Args:
        - dataframe: The dataframe to test.
    
    Returns: True if the dataset can be merged. False otherwise.
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
        - merged: The dictionary containing the merged dataset.
        - config: The configuration file of this program execution.

    Returns: A dataframe of the merged datasets. This list can be saved in a file with the
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

def monitor_dataset(reference: DataFrame, merged: DataFrame) -> DataFrame:
    """
    Creates a dataframe that tracks changes between two program executions.

    Pandas manages this dataframe. This minimal dataframe shows column and value changes.

    Args:
        - reference: The reference dataframe, corresponding to when the first data extraction was
            executed.
        - merged: The data frame of this current program execution.
    
    Returns: The compared data frame from Pandas.
    """
    monitered = reference.compare(merged, result_names=('reference', 'new'))
    return monitered

def get_observations_with_complete_years(merged_dataframe: DataFrame) -> DataFrame:
    """
    Filters a dataframe so that all countries have an observation for a given year. If a country has
    one null value for this year, all the observations for said years are removed.

    Args:
        - merged_dataframe: The dataframe for the merged indicators.
    
    Returns: The dataframe that satisfies the filter criterion, where each year includes all
        countries with observations for all indicators.
    """

    complete_observations = merged_dataframe[~merged_dataframe.Country.str.contains('European Union')].dropna()
    years = complete_observations.Year.unique()
    number_of_countries = len(complete_observations.Country.unique())

    for year in years:
        if len(complete_observations[complete_observations.Year == year]) != number_of_countries:
            complete_observations = complete_observations[complete_observations.Year != year]

    return complete_observations

def get_years_to_compute(complete_observations: DataFrame) -> List[float]:
    """
    For a dataframe with complete observations, it gets the years to compute a sensitivity for
    observation years in the PCA. Three years are used: the first, the middle and the last.

    Args:
        - complete_observations: The dataframe to compute the years from. It should have a `Year`
            column.

    Returns: The three years to compute a sensitivity from. The selected years are the first
        observed values, the last observed values and the middle value.
    """

    complete_years = complete_observations.Year.unique()
    middle_index = round((len(complete_years) - 1) / 2)
    return [
        complete_years[0].item(),
        complete_years[middle_index].item(),
        complete_years[-1].item()
    ]
