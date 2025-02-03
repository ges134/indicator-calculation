"""
This module ensures that all the indicators' datasets are merged into one.
"""

from typing import List, Dict
from tqdm import tqdm
from pandas import DataFrame

from data import load_dataset

def merge_datasets(codes: List[str]) -> Dict:
    """
    This function applies the dataset merging from the `codes` provided
    in the parameter. This is the central function of merging.

    This function is still in development. Once done, the return type will be
    a list of dictionaries. For the moment, it returns which dataset that
    will be computed and merged.

    Each dataset is computed entirely before moving on to the next one. The
    computation does the following elements:
    
    1. Load the dataset and convert it into a `DataFrame`
    2. Check if the dataset is of appropriate format. If not, the computing
    stops there for this dataset.
    3. With the appropriate format, it scans each row and updates the merged
    dataset. New keys are created if they don't exists.

    Args:
        codes: The dataset codes to be merged into one dataset.
    
    Returns:
        A list of flags indicating which dataset will be computed and merged.
        The order follows the same order as the `codes` parameters. This is
        a temporary return to test data parsing.
    """

    merged = {}

    for code in tqdm(
        codes,
        'Computation of datasets',
        leave=False
    ):
        dataframe = load_dataset(code)
        if dataset_can_be_merged(dataframe):
            unit_of_measure = dataframe['Unit of measure'][0]
            for row in tqdm(
                dataframe[dataframe['Unit of measure'] == unit_of_measure].itertuples(),
                f'Merging dataset {code}',
                leave=False
            ):
                geopolitical_column_location = dataframe.columns.get_loc(
                    'Geopolitical entity (reporting)'
                )
                row_key = f'{row[geopolitical_column_location+1]}-{row.Time}'
                if row_key not in merged:
                    merged[row_key]= {
                        'values': {}
                    }

                merged[row_key]['values'][code] = row.value

    return merged

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
