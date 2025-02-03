"""
The data module ensures data manipulation. It reads files or dataset and saves
the results.
"""

from pyjstat.pyjstat import Dataset
from pandas import DataFrame
from typing import List, Union
from csv import writer

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

def save_merged_dataset(data: List[List[Union[str, float]]]):
    """
    Saves the CSV-ready merged dataset into the `data/` folder.

    The saved file will be available in `data/merged.csv`.

    Args:
        data: The CSV data. Use the `merger.merged_dataset_to_csv` method to generate the
            adequate data.
    """
    with open('./data/merged.csv', 'w', encoding='utf-8', newline='') as file_stream:
        csv_writer = writer(file_stream)
        csv_writer.writerows(data)
