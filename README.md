# Indicator collection

The indicator collection program queries Eurostat databases for selected indicator datasets and then merges them into a single file for results analysis.

The README page gives information on the program structure and usage. It stats with a get-started guide and a usage guide. This page then presents the program flow.

## Getting started

The project was build using [Pyton 3.12](https://www.python.org/) and [Pipenv](https://pypi.org/project/pipenv/). Both should be installed before the first usage of this program.

The program will need a `codes.txt` file in the `data/` repository. There should be one code per line in the file. The codes should also be available in the statistics API of Eurostat. A `codes.template.txt` file is given as a starting point to use the program.

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
1. The dataset is tested for merging conditions. A dataset can be merged if it has a annual time stamp, a geopolitical entity, a unit of measure and a time.

Further developments will give more instructions in the program flow.
