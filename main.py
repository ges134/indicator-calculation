"""
The main module contains the main program of the indicator calculation.

A typical program execution will query Eurostat databases, merge the collected datasets into one
dictionnary and compute elements necessary for the integrated subjective-objective approach. Note
that only the degree of independance for aggregation and the subjective approach is computed by the
program. The remainder should be set up in a spreadsheet software. All generated data by the program
is saved into a file.
"""

from pandas import DataFrame

from merger import merge_datasets, convert_dataset_to_dataframe
from data import load_config, save_csv
from independance import (
  apply_pca_on_indicators,
  get_degrees_of_independance,
  prepare_dataframe_for_pca
)

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
    merged_dataframe = convert_dataset_to_dataframe(merged, config)
    codes = [c['id'] for c in config]

    print('Preparing indicators for PCA')
    indicators_data = prepare_dataframe_for_pca(merged_dataframe)

    print('Computing PCA on indicators')
    eigen_values, eigen_vectors, explained_variance = apply_pca_on_indicators(indicators_data)

    eigen_values_dataframe = DataFrame(
        eigen_values.reshape(1, 6),
        columns=[f'PC {i+1}' for i in range(len(eigen_values))]
    )

    eigen_vectors_dataframe = DataFrame(
        eigen_vectors,
        columns=[f'PC {i+1}' for i in range(len(eigen_vectors))]
    )

    explained_variance_dataframe = DataFrame(
        [explained_variance],
        columns=[f'PC {i+1}' for i in range(len(explained_variance))]
    )

    print('Computing degrees of independance')
    angle_matrix, independance_matrix = get_degrees_of_independance(eigen_vectors)

    angle_matrix_dataframe = DataFrame(angle_matrix, columns=codes)
    independance_matrix_dataframe = DataFrame(independance_matrix, columns=codes)


    print('Saving files')
    save_csv(merged_dataframe, 'merged.csv')
    save_csv(eigen_values_dataframe, 'eigen-values.csv')
    save_csv(eigen_vectors_dataframe, 'eigen-vectors.csv')
    save_csv(explained_variance_dataframe, 'explained-variance.csv')
    save_csv(angle_matrix_dataframe, 'angles.csv')
    save_csv(independance_matrix_dataframe, 'independance_degree.csv')


if __name__ == '__main__':
    main()
