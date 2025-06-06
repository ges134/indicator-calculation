"""
The data module ensures data manipulation. It reads files or dataset and saves
the results.
"""

from pyjstat.pyjstat import Dataset
from pandas import DataFrame
from typing import List, Dict
from json import loads
from requests import get

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
    response = get(url, timeout=30)
    dataset = Dataset.read(response.text)
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

def load_config() -> List[Dict]:
    """
    This functtion loads the config file that is stored locally. The config file should be provided
    in the `data/repository`.

    Returns:
        The loaded configuration file.
    """
    return loads(load_file('config.json'))

def save_csv(dataframe: DataFrame, filepath: str, index=False):
    """
    Saves a dataframe into the `data/` folder.

    Args:
        data: The Dataframe to be converted into a CSV File.
        filepath: The path of the file, relative to the data repository.
        If a file is in a subdirectory, the subdirectory should also be included.
        The `.csv` extension should be included in the file path.
        ìndex: Whether or not to preserve the row number in the saved CSV. The default value is
        `False`.
    """
    dataframe.to_csv(f'./data/{filepath}', index=index)
