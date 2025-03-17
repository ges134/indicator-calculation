# Indicator calculation

The indicator calculation program queries Eurostat databases for selected indicator datasets, merges them into a single file for results analysis and then computes some elements of the integrated objective-subjective approach. The elements computed are the degree of indenpendance and the AHP.

The README page gives information on the program structure and usage. It stats with a get-started guide and a usage guide. This page then presents the program flow.

## Getting started

The project was build using [Pyton 3.12](https://www.python.org/) and [Pipenv](https://pypi.org/project/pipenv/). Both should be installed before the first usage of this program.

The program will need a `config.json` file in the `data/` repository. A sample `config.template.json` file is given as a starting point to use in the program. The configuration file is a list of objects. Each object has the following structure:

```json
{
  "id": "internal id",
  "code": "eurostat code"
  // A list of key-values for dimensions.
}
```

Where:

- `id` (**Mandatory**): The identifier of the indicator to be collected. This identifier is internal and will be found again in the merged file.
- `code` (**Mandatory**): The Eurostat identifier of this indicator.
- `key-values-dimensions`: Each indicator should specify the different dimensions of data when there is more than one available option. It can be omitted when there is only one dimension. For instance, if the indicator reports multiple units of measures, the configuration file should have a key value `"Unit of measure": "Unit"`. On the Eurostat website, the _Customize your dataset_ shows the different dimensions. The dimensions _Time_, _Time Frequency_ and _Geopolitical entity (reporting)_ should be omitted too as they are checked by default in the program. If these columns are not available, the dataset will not be merged.

To run the project, the dependencies should be installed first with the following command:

```sh
pipenv install
```

## Usage

The program can be invoked from a command line and should be invoked with the source code. The command follows the following format:

```sh
pipenv run py main.py
```

## Program flow

The program flow will parse each indicator in the `codes` file. Each indicator gets the following treatment:

1. The dataset is loaded from the Eurostat database and converted into a `Dataframe`.
1. The dataset is tested for merging conditions. A dataset can be merged if it has a annual time stamp, a geopolitical entity and a time.
1. The dataset is filtered to have a single dimension for every other dimensions according to the configuration file. For instance, if there is multiple units of measure, the program will use the one specified in the configuration file.
1. With the appropriate format, it scans each row and updates the merged dataset. New keys are created if they don't exists.

The created dataset is then converted into a PCA-ready dataset. The program then does the following with the dataset:

1. Compute the PCA for the indicators
1. Compute the degrees of independance of the indicators.

A degree of independance is a value between 0 and 1 that shows how a pair of indicators is independant from one another. This is made from the application that PC are uncorollated. Hence, the closer an indicator is to the value of 0, the more independant they are.

## Saved data

The program saves multiple results data. They are detailed below:

- `angles.csv`: The angles computed between each indicators.
- `eigen-values.csv`: The eigen values of each computed principal component.
- `eigen-vectors.csv`: The eigen vectors of each computed principal component.
- `explained-variance.csv`: The explained variance of each computed principal componenent.
- `independance_degree.csv`: The degree of independance between each indicators.
- `merged.csv`: The merged dataset.

The angles and independances degree are in triangular format. Anything under the diagonal is unused.
