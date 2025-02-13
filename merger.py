"""
This module ensures that all the indicators' datasets are merged into one.
"""

from typing import List, Dict, Union, Tuple
from tqdm import tqdm
from pandas import DataFrame
from math import isnan

from data import load_dataset

def merge_datasets(codes: List[str], units: Dict) -> Tuple[Dict, Dict]:
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
        codes: The dataset codes to be merged into one dataset.
        units: A dictionnary of units of measures that were preselected. If, for a given code, there
            is a unit of measure in the `units` dictionnary, the method will use the specified unit
            of measure. Otherwise, the first unit of measure reported in the dataset will be used. A
            compiled list of units of measure is returned by this method.
    
    Returns:
        A tuple containing the merged datasets and the compiled units of measure. In the merged
        dataset dictionnary, each entry has a key with the country and the year of reporting, 
        separated by a semi-colomn (;). Inside, another dictionnary with the values is stored. The
        values can be accessed with the `value` key. In this dictionnary, the indicator code is the 
        key and the indicator value is in the value. In the compiled units of measure dictionnary,
        each each entry is the code of the indicator and the value represents the reported unit of
        measure. If the unit was specified by the `units` argument, this value will remain the same
        in this dictionnary.
    """

    merged = {}
    units_final = units.copy()

    for code in tqdm(
        codes,
        'Computation of datasets',
        leave=False
    ):
        dataframe = load_dataset(code)
        if dataset_can_be_merged(dataframe):
            unit_of_measure = units[code] \
                if code in units_final \
                else dataframe['Unit of measure'][0]

            units_final[code] = unit_of_measure
            for row in tqdm(
                dataframe[dataframe['Unit of measure'] == unit_of_measure].itertuples(),
                f'Merging dataset {code}',
                leave=False
            ):
                geopolitical_column_location = dataframe.columns.get_loc(
                    'Geopolitical entity (reporting)'
                )
                row_key = f'{row[geopolitical_column_location+1]};{row.Time}'
                if row_key not in merged:
                    merged[row_key]= {
                        'values': {}
                    }

                merged[row_key]['values'][code] = row.value

    return merged, units_final

def dataset_can_be_merged(dataframe: DataFrame) -> bool:
    """
    This function checks if the data frame given in the `dataframe` parameter 
    can be merged. A data frame can be merged if there is a annual time frequency
    and if the unit of measure, geopolitical entity and time columns are on the dataset.

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
        'Unit of measure' not in dataframe.columns
        or 'Geopolitical entity (reporting)' not in dataframe.columns
        or 'Time' not in dataframe.columns
    ):
        return False
    return True

def merged_dataset_to_csv(merged: Dict, codes: List[str]) -> List[List[Union[str, float]]]:
    """
    Converts the merged dataset into a CSV-ready list for data saving.

    The merged dataset can be generated with the `merge_datasets` method of this module.

    Args:
        merged: The dictionnary containing the merged dataset.
        codes: The dataset codes to be merged into one dataset.

    Returns:
        A CSV-ready list of the merged datasets. This list can be saved into a file with the
        `data.save_csv` method.
    """

    parsed_csv = [['Country', 'Year'] + codes]

    for key, entry in tqdm(
        merged.items(),
        'Conversion of dataset into a CSV file',
        leave=False
    ):
        country, year = key.split(';')
        row = [country, year] + [''] * len(codes)
        for indicator, value in entry['values'].items():
            index = codes.index(indicator)
            row[index+2] = '' if isnan(value) else value

        parsed_csv.append(row)

    return parsed_csv

def units_to_csv(units: Dict) -> List[List[str]]:
    """
    Converts the compiled units of measure into a CSV-ready list for data saving.

    The compiled units of measure can be generated with the `merge_datasets` method of this module.

    Args:
        units: The dictionnary containing the compiled units.

    Returns:
        A CSV-ready list of the compiled units of measure. This list can be saved into a file with
        the `data.save_csv` method.
    """

    parsed_csv = [['Indicator code', 'Unit of measure']]
    for code in tqdm(
        units,
        'Conversion of units into a CSV file',
        leave=False
    ):
        parsed_csv.append([code, units[code]])

    return parsed_csv
