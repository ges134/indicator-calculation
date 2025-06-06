"""
The monitor module contains a subprogram to monitor data changes throughout the study of the
indicators.

This will merge collected datasets and compare them to a reference dataset. The reference dataset
should be supplied. The difference is saved into a CSV file. If the file is empty, it means that no
data differences were noted. The remainder of the work can be set up in a spreadsheet software.
"""

from pandas import read_csv

from data import load_config, save_csv
from merger import merge_datasets, convert_dataset_to_dataframe, monitor_dataset

def main():
    """
    Main execution of the subprogram.
    """

    print('Indicator data calculation program')
    print('This subprogram looks for changes of the considered datasets')

    print('Reading reference file')
    reference = read_csv('./data/reference.csv')

    print('Reading configuration')
    config = load_config()

    print('Merging datasets')
    merged = merge_datasets(config)
    merged_dataframe = convert_dataset_to_dataframe(merged, config)

    print('Comparing datasets')
    monitored = monitor_dataset(reference, merged_dataframe)

    print('Saving files')
    save_csv(monitored, 'monitored.csv', True)

if __name__ == '__main__':
    main()
