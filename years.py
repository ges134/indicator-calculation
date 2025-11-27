"""
The years module contains a subprogram to compute sensitivity based on the observed years for the
PCA. It computes only the necessary data to perform the sensitivity analysis.

This program merges the collected dataset and filters any year in which at least one country has a
null observation for one indicator. Then, it takes three years among the complete variables, the
first, the middle and the last, and it computes the PCA and the degree of independence. Those are
saved into a CSV file.
"""

from pandas import DataFrame

from data import load_config, save_csv
from merger import (
    merge_datasets,
    convert_dataset_to_dataframe,
    get_observations_with_complete_years,
    get_years_to_compute
)
from independance import (
    get_pca_data_from_years,
    get_degrees_of_independance
)
from stats import apply_pca

def main():
    """
    Main execution of the subprogram.
    """

    print('Indicator data collection program')
    print('This subprograms obtains data for the observation years sensitivity analysis')

    print('Reading configuration')
    config = load_config()
    codes = [c['id'] for c in config]

    print('Obtaining datasets')
    merged = merge_datasets(config)
    merged_dataframe = convert_dataset_to_dataframe(merged, config)
    complete_observations = get_observations_with_complete_years(merged_dataframe)
    years_to_compute = get_years_to_compute(complete_observations)

    print('Obtaining result data')
    for year in years_to_compute:
        pca_data = get_pca_data_from_years(complete_observations, year)
        _, eigen_vectors, _ = apply_pca(pca_data)
        _, independance_matrix = get_degrees_of_independance(eigen_vectors)
        eigen_vectors_dataframe = DataFrame(
            eigen_vectors,
            columns=[f'PC {i+1}' for i in range(len(eigen_vectors))]
        )
        independance_matrix_dataframe = DataFrame(independance_matrix, columns=codes)
        save_csv(eigen_vectors_dataframe, f'{year}-eigen-vectors.csv')
        save_csv(independance_matrix_dataframe, f'{year}-independance-degree.csv')

if __name__ == '__main__':
    main()
