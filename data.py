"""
The data module ensures data manipulation. It reads files or dataset and saves
the results.
"""

from pyjstat.pyjstat import Dataset
from pandas import DataFrame
from typing import List, Union, Dict
from csv import writer

from tqdm import tqdm

def load_dataset(code: str) -> DataFrame:
    """
    This function loads a dataset from Eurostat and gives a dataframe with
    the parsed data.

    This is mainly managed by the `pyjstat` library.

    Args:
        code: the indicator code to fetch. This code should come from the 
        Eurostat database.

    Returns:
        The parsed dataframe with the data of the given indicator.
    """
    url = f'https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{code}'
    dataset = Dataset.read(url)
    return dataset.write('dataframe')

def load_file(filepath: str) -> str:
    """
    This function loads a file stored locally. Files should be in the `data/`
    repository.

    This is used to load the indicator codes. It is also used for testing purposes.

    Args:
        filepath: The path of the file, relative to the data repository.
        If a file is at the `data/` repository, the file name can be inputted.
        If a file is in a subdirectory, the subdirectory should also be included.
        For the tests files, this means that the `filepath` argument would look like
        `tests/testfile.txt`.
    """
    with open(f'./data/{filepath}', encoding='utf-8') as file:
        return file.read()

def save_csv(data: List[List[Union[str, float]]], filepath: str):
    """
    Saves a CSV-ready variable into the `data/` folder.

    Args:
        data: The CSV data ready to be merged.
        filepath: The path of the file, relative to the data repository.
        If a file is in a subdirectory, the subdirectory should also be included.
        The `.csv` extension should be included in the file path.
    """
    with open(f'./data/{filepath}', 'w', encoding='utf-8', newline='') as file_stream:
        csv_writer = writer(file_stream)
        csv_writer.writerows(data)

def transform_units_from_file(units_raw: str) -> Dict:
    """
    Transforms the units of measure file into a dictionnary used to compile units of measures.

    Args:
        units_raw: The read file of the units of measure files.

    Returns:
        A dictionnary with the selected unit of measures. Each key in this dictionnary represents
        the code of an indicator and the value represents the selected unit of measure for this
        indicator.
    """

    units_line = [u for u in units_raw.split('\n') if u != '']
    units = {}
    for line in tqdm(
        units_line,
        'Conversion of units',
        leave=False
    ):
        indicator, unit = line.split(' UNIT ')
        units[indicator] = unit

    return units
