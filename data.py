"""
The data module ensures data manipulation. It reads files or datasets and saves the results.
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

     The `pyjstat` library mainly manages this method.

    Args:
        - code: the indicator code to fetch. This code should come from the Eurostat database.

    Returns: The parsed dataframe with the data of the given indicator.
    """
    url = f'https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{code}'
    response = get(url, timeout=30)
    dataset = Dataset.read(response.text)
    return dataset.write('dataframe')

def load_file(filepath: str) -> str:
    """
    This function loads a local file. Files should be in the `data/` repository.

    Args:
        - filepath: The path of the file, relative to the data repository. If a file is in the
        `data/` repository, its filename can be entered. If a file is in a subdirectory, the
        subdirectory should also be included. For the test files, this means that the `filepath`
        argument would look like `tests/testfile.txt`.
    """
    with open(f'./data/{filepath}', encoding='utf-8') as file:
        return file.read()

def load_config() -> List[Dict]:
    """
    This function loads the locally stored configuration file. The configuration file should be
    provided in the `data/repository`.

    Returns: The loaded configuration file.
    """
    return loads(load_file('config.json'))

def save_csv(dataframe: DataFrame, filepath: str, index=False):
    """
    Saves a dataframe into the `data/` folder.

    Args:
        - data: The Dataframe to be converted into a CSV File.
        - filepath: The path of the file, relative to the data repository. If a file is in a 
            subdirectory, the subdirectory should also be included. The `.csv` extension should be
            included in the file path.
        - index: Whether or not to preserve the row number in the saved CSV. The default value is
            `False`.
    """
    dataframe.to_csv(f'./data/{filepath}', index=index)
