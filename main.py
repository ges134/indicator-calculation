"""
The main module contains the main program of the indicator collector.

A typical program execution will query Eurostat databases and merge the
collected datasets into one file. The program will then save the file.
"""

from merger import merge_datasets
from data import load_file

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

    print('Merging datasets')
    merged = merge_datasets(codes)

    print('Merged dataset')
    print('This dictionnary will not be displayed in a next version of the program')
    print(merged)

if __name__ == '__main__':
    main()
