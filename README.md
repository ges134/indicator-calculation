# Indicator calculation

The indicator calculation program queries Eurostat databases for selected indicator datasets, merges them into a single file for results analysis and then computes some elements of the integrated objective-subjective approach. The elements computed are the degree of indenpendance and the AHP.

The README page gives information on the program structure and usage. It stats with a get-started guide and a usage guide. This page then presents the program flow.

## Getting started

The project was build using [Pyton 3.12](https://www.python.org/) and [Pipenv](https://pypi.org/project/pipenv/). Both should be installed before the first usage of this program.

The program will need a `config.json` file in the `data/` repository. A sample `config.template.json` file is given as a starting point to use in the program. The configuration file is a list of objects. Each object has the following structure:

```json
{
  "id": "internal id",
  "code": "eurostat code",
  "social": 0,
  "environmental": 0,
  "economic": 0
  // A list of key-values for dimensions.
}
```

Where:

- `id` (**Mandatory**): The identifier of the indicator to be collected. This identifier is internal and will be found again in the merged file.
- `code` (**Mandatory**): The Eurostat identifier of this indicator.
- `social`: The score, between 0 and 3, for the social pillar for this indicator.
- `environmental`: The score, between 0 and 3, for the environmental pillar for this indicator.
- `economic`: The score, between 0 and 3, for the economic pillar for this indicator.
- `key-values-dimensions`: Each indicator should specify the different dimensions of data when there is more than one available option. It can be omitted when there is only one dimension. For instance, if the indicator reports multiple units of measures, the configuration file should have a key value `"Unit of measure": "Unit"`. On the Eurostat website, the _Customize your dataset_ shows the different dimensions. The dimensions _Time_, _Time Frequency_ and _Geopolitical entity (reporting)_ should be omitted too as they are checked by default in the program. If these columns are not available, the dataset will not be merged.

The scores are on a scale of one to three with a particular significance. The scores can be interpreted as such:

- 0: The indicator does not cover this pillar.
- 1: The indicator covers the pillar, but it is not a considered as the main focus of the indicator.
- 2: The indicator covers the pillar and shares its main focus with another pillar.
- 3: The indicator covers the pillar and its the main focus.

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

The program then reuses the configuration file to simulate an AHP process and then gain subjective weights. The program does the following with the configuration file:

1. Convert the scores into a five points Likert Scale.
1. Use the Likert scale to compute comparison matrices.
1. Apply AHP for each pillars and for the pillar themselves with consistency analysis.

The Likert scale can also be interpreted with the following rules:

- 1: The indicator has a difference of at least two points with the pillar.
- 3: The indicator has a difference of one point with the pillar.
- 5: The indicator has the same score as the pillar. A difference of 3 or 4 point is observed for the other pillars.
- 7: The indicator has the same score as the pillar. A difference of 1 or 2 point is observed for the other pillars.
- 9: The indicator is a perfect match with the pillar.

## Saved data

The program saves multiple results data and one figure. They are detailed below:

- `angles.csv`: The angles computed between each indicators.
- `consistency.csv`: The consistency analysis for this program execution, per pillar and with the pillars.
- `contribution.png`: The contribution graph of the PCA.
- `economic-comparison-matrix.csv`: The comparison matrix for the economic sustainability pillar.
- `eigen-values.csv`: The eigen values of each computed principal component.
- `eigen-vectors.csv`: The eigen vectors of each computed principal component.
- `environmental-comparison-matrix.csv`: The comparison matrix for the environmental sustainability pillar.
- `explained-variance.csv`: The explained variance of each computed principal componenent.
- `independance_degree.csv`: The degree of independance between each indicators.
- `merged.csv`: The merged dataset.
- `scores.csv`: The scores of each indicator in the Likert scale.
- `social-comparison-matrix.csv`: The comparison matrix for the social sustanibility pillar.
- `weights.csv`: The weights of each indicator according to each pillar and the final weights of said indicators.

The angles and independances degree are in triangular format. Anything under the diagonal is unused.

## Monitoring data changes

A subprogram monitors changes in the indicators data in the Eurostat database. To monitor said changes, the configuration file and a reference merged dataset must be provided. The reference merged dataset is the `merged.csv` file associated to the configuration of the program. The file must be named `reference.csv` in order for the subprogram to work. The following command can run the program:

```sh
pipenv run py monitor.py
```

The program will do the following.

1. Load the reference dataset and load the dataset with Eurostat data.
1. Create a Pandas `DataFrame` that compares old and new values.
1. Save said dataset.

The saved file is named `monitored.csv` and follows the [Pandas `compare` format](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.compare.html). In summary, if the file is empty, no changes were reported. If there is a change, the dataset will only reflect those changes. Hence, not all indicators and not all rows may be present in the monitored file. The changed indicators will appear with the indicator identifier and two column, `reference` and `new`. The former referes to the reference dataset, provided with the execution program, while the latter is the one that is associated with the current program execution. The row number appears as the first column, which can be used to track changes. `NaN` may appear to indicate that no changes were observer for this indicator in this row.
