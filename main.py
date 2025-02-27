"""
The main module contains the main program of the indicator calculation.

A typical program execution will query Eurostat databases, merge the collected datasets into one
dictionnary and compute elements necessary for the integrated subjective-objective approach. Note
that only the degree of independance for aggregation and the subjective approach is computed by the
program. The remainder should be set up in a spreadsheet software. All generated data by the program
is saved into a file.
"""

from merger import merge_datasets, merged_dataset_to_csv
from data import load_config, save_csv

def main():
    """
    Main execution of the program.
    """

    print('Indicator data collection program')
    print('This program collects all datasets need to complete the indicator axis')
    print('This program is still under development')

    print('Reading configuration')
    config = load_config()

    print('Merging datasets')
    merged = merge_datasets(config)

    print('Saving merged dataset')
    merged_csv = merged_dataset_to_csv(merged, config)
    save_csv(merged_csv, 'merged.csv')

if __name__ == '__main__':
    main()
