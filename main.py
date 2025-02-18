"""
The main module contains the main program of the indicator calculation.

A typical program execution will query Eurostat databases, merge the collected datasets into one
dictionnary and compute elements necessary for the integrated subjective-objective approach. Note
that only the degree of independance for aggregation and the subjective approach is computed by the
program. The remainder should be set up in a spreadsheet software. All generated data by the program
is saved into a file.
"""

from merger import merge_datasets, merged_dataset_to_csv, units_to_csv
from data import load_file, save_csv, transform_units_from_file

def main():
    """
    Main execution of the program.
    """

    print('Indicator data collection program')
    print('This program collects all datasets need to complete the indicator axis')
    print('This program is still under development')

    print('Reading indicator codes')
    codes_raw = load_file('codes.txt')
    codes = [c for c in codes_raw.split('\n') if c != '']

    print('Reading default unit of measures')
    units_raw = load_file('units.txt')
    units = transform_units_from_file(units_raw)

    print('Merging datasets')
    merged, units = merge_datasets(codes, units)

    print('Saving merged dataset')
    merged_csv = merged_dataset_to_csv(merged, codes)
    save_csv(merged_csv, 'merged.csv')

    print('Saving units of measure')
    units_csv = units_to_csv(units)
    save_csv(units_csv, 'units.csv')

if __name__ == '__main__':
    main()
