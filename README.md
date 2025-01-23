# Indicator collection

The indicator collection program queries Eurostat databases for selected indicator datasets and then merges them into a single file for results analysis.

The README page gives information on the program structure and usage. It stats with a get-started guide and a usage guide. This page then presents the program flow.

## Getting started

The project was build using [Pyton 3.12](https://www.python.org/) and [Pipenv](https://pypi.org/project/pipenv/). Both should be installed before the first usage of this program.

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

A typical usage will query eurostat databases. Then, it will merge all the datasets into a single one.
