# Computation program of A novel integrated objective-subjective weighting method using principal component analysis to correct correlation bias among criteria in sustainability decision making

This computation program queries the Eurostat database for selected indicator datasets, merges them into a single dataset, and then computes elements of the integrated objective-subjective approach detailed in the paper. The following elements are computed:

- Principal component analysis (PCA) with degrees of independence
- Confidence intervals of the PCA
- Subjective weighting

This software and its source code are given as supplementary material from the paper _A novel integrated objective-subjective weighting method using principal component analysis to correct correlation bias among criteria in sustainability decision making_ with the following complete reference:

{{ REFERENCE to be added }}

The data used in the paper, which was computed with this program, is available as a dataset with the following complete reference:

Anglehart-Nunes, Jessy; Glaus, Mathias, 2025, "Dataset of A novel integrated objective-subjective weighting method using principal component analysis to correct correlation bias among criteria in sustainability decision making", https://doi.org/10.5683/SP3/NYKQXH, Borealis, VERSION PROVISOIRE, UNF:6:IWrDTTuFjH5uetnHcQTp/Q== [fileUNF]

The README page gives information on the program usage. It starts with a get-started guide and a usage guide. This page then presents the program flow and the created subprograms.

## Getting started

The project was built using [Python 3.12](https://www.python.org/) and [Pipenv](https://pypi.org/project/pipenv/). Both should be installed before the first usage of this program.

The program will need a `config.json` file in the `data/` repository. A sample `config.template.json` file is provided as a starting point for the program. The `ProgramConfiguration` files are also given for replication purposes.

The configuration file is a list of objects. Each object has the following structure:

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

- `id` (**mandatory**): The identifier of the indicator to be collected. This identifier is internal and will be found again in the merged file.
- `code` (**mandatory**): The Eurostat identifier of this indicator.
- `social` (**mandatory**): The score, between 0 and 3, for the social pillar for this indicator.
- `environmental` (**mandatory**): The score, between 0 and 3, for the environmental pillar for this indicator.
- `economic` (**mandatory**): The score, between 0 and 3, for the economic pillar for this indicator.
- `key-values-dimensions`: Each indicator should specify the different dimensions of data when there is more than one available option. It can be omitted when there is only one dimension. For instance, if the indicator reports multiple units of measure, the configuration file should include the key-value pair `"Unit of measure": "Unit"`. On the Eurostat website, the _Customize your dataset_ shows the different dimensions. The dimensions _Time_, _Time Frequency_ and _Geopolitical entity (reporting)_ should also be omitted, as they are checked by default in the program. If these columns are not available, the dataset will not be merged.

The scores are on a scale of 0 to 3 with a particular significance. The scores can be interpreted as such:

- 0: This indicator does not mention the sustainable development pillar.
- 1: This indicator mentions the sustainable development pillar.
- 2: The sustainable development pillar is one of the primary focuses of this indicator.
- 3: The sustainable development pillar is the sole primary focus of this indicator

To run the project, the dependencies should be installed first with the following command:

```sh
pipenv install
```

## Usage

The program can be invoked from a command line and must be invoked with the source code. The command is as follows:

```sh
pipenv run py main.py
```

## Program flow

The program flow will parse each indicator in the `config.json` file. Each indicator gets the following treatment:

1. The dataset is loaded from the Eurostat database and converted into a `Dataframe`.
1. The dataset is tested for merging conditions. A dataset can be merged if it has an annual timestamp, a geopolitical entity, and a time.
1. The dataset is filtered to have a single dimension from the configuration file. For instance, if multiple units of measure are specified, the program will use the one specified in the configuration file
1. With the appropriate format, it scans each row and updates the merged dataset.

The created dataset is then converted into a PCA-ready dataset. The program then does the following with the dataset:

1. Compute the PCA for the indicators.
1. Compute the degrees of independence of the indicators.

A degree of independence is a value between 0 and 1 that indicates how a pair of indicators is independent of one another. 0 indicates total dependence, while 1 indicates total independence.

Afterwards, the program computes confidence intervals based on the bootstrap method. It implies the following steps:

1. Normalize the dataset that does not follow a normal distribution.
1. Draw bootstrap samples of the normalized indicators and apply the PCA to these bootstraped samples.
1. Create the jacknifed dataset of the normalized indicators and apply the PCA to these jacknifed samples.
1. Produce the confidence intervals for the significant levels 0.01 and 0.05.

The program then reuses the configuration file to simulate an Analytic Hierarchy Process (AHP) to compute subjective weights. The program does the following computations:

1. Convert the scores into a five-point Likert Scale.
1. Use the Likert scale to compute comparison matrices.
1. Apply AHP for each pillar and for the pillars themselves with consistency analysis.

The Likert scale can also be interpreted with the following rules:

- 1: The indicator has a score of 0 or 1 for the pillar
- 3: The indicator has a score of 2 for the pillar
- 5: The indicator has a score of 3 for the pillar. A 2-point difference is observed for the other pillars.
- 7: The indicator has a score of 3 for the pillar. A 1-point difference is observed for the other pillars.
- 9: The indicator is a perfect match with the pillar.

## Saved data

The program saves multiple results. They are detailed below:

- `angles.csv`: The angles computed between each indicator.
- `bootstraped-dataset.csv`: The dataset of the bootstrap samples.
- `confidence-intervals-01.csv`: The values of the bounds for the confidence intervals with the significance level 0.01.
- `confidence-intervals-05.csv`: The values of the bounds for the confidence intervals with the significance level 0.05.
- `consistency.csv`: The consistency analysis for this program execution, per pillar and with the pillars.
- `economic-comparison-matrix.csv`: The comparison matrix for the economic sustainability pillar.
- `eigen-values.csv`: The eigenvalues of each computed principal component.
- `eigen-vectors.csv`: The eigenvectors of each computed principal component.
- `empiric-eigen-vectors.csv`: The eigenvectors of each computed principal component with data normalization for bootstrapping.
- `environmental-comparison-matrix.csv`: The comparison matrix for the environmental sustainability pillar.
- `explained-variance.csv`: The explained variance of each computed principal component.
- `independance_degree.csv`: The degree of independence between each indicator.
- `jacknifed-dataset.csv`: The dataset of the jacknifed data.
- `merged.csv`: The merged dataset.
- `scores.csv`: The scores of each indicator in the Likert scale.
- `social-comparison-matrix.csv`: The comparison matrix for the social sustainability pillar.
- `weights.csv`: The weights of each indicator according to each pillar and the final weights of said indicators.

The angles and independence degrees are in triangular format. Anything under the diagonal is unused. Furthermore, while this program saves contribution graphs, the ones used in the paper are available as notebooks in the `notebooks/` directory.

## Sensitivity for the observed years

A subprogram allows the application of PCA and the computation of independence degrees for three observed years with complete data: the first, the middle, and the last. To use this subprogram, the configuration file must be provided. The following command can run the program:

```sh
pipenv run py years.py
```

The program will do the following;

1. Load the configuration
1. Obtain the dataset for the indicators
1. Filter non-complete years
1. Compute PCA and degrees of independence for three years.
1. Save computed files.

The saved files are `YEAR-eigen-vectors.csv` and `YEAR-independance-degree.csv`, which have the same format as the main execution program, but the results were computed with the given YEAR.

## Monitoring data changes

A subprogram monitors changes in the indicators data in the Eurostat database. To monitor said changes, the configuration file and a reference merged dataset must be provided. The reference merged dataset is the `merged.csv` file used by the program's configuration. The file must be named `reference.csv`. The following command can run the subprogram:

```sh
pipenv run py monitor.py
```

The program will do the following:

1. Load the reference dataset and load the dataset with Eurostat data.
1. Create a Pandas `DataFrame` that compares old and new values.
1. Save said dataset.

The saved file is named `monitored.csv` and follows the [Pandas `compare` format](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.compare.html). In summary, if the file is empty, no changes were reported. If there is a change, the dataset will only reflect those changes. Hence, not all indicators or rows may be present in the monitored file. The changed indicators will appear with the indicator identifier and two columns, `reference` and `new`. The former refers to the reference dataset provided with the execution program, while the latter is the dataset associated with the current program execution. The row number appears as the first column, which can be used to track changes. `NaN` may appear to indicate that no changes were observed for this indicator in this row.
