"""
The scripts run linting on all the program's files.

The script fails if the pylint score is below a defined threshold.
"""

import sys
from pylint import lint

THERESHOLD = 9.5
DISABLE = '--disable=too-many-arguments,wrong-import-order,redefined-builtin'


def lint_component():
    """
    Checks if all the files follow the linting guidelines from `pylint`.
    """

    print('Running linter...')

    files = [
        'main.py',
        'lint.py',
        'merger.py',
        'data.py',
        'independance.py',
        'subjective.py',
        'contribution.py',
        'monitor.py',
        'stats.py',
        'confidence.py',
        'years.py',
        'tests/constants.py',
        'tests/test_merger.py',
        'tests/test_independance.py',
        'tests/test_subjective.py',
        'tests/test_contribution.py',
        'tests/test_stats.py',
        'tests/test_confidence.py',
    ]
    scores = []

    for file in files:
        print(f'Running linter on file: {file}')
        run = lint.Run([file, DISABLE], None, False)
        scores.append(run.linter.stats.global_note)

    print('Linter execution complete!')
    print('Performing checks')

    has_failed = False
    for i, file in enumerate(files):
        if scores[i] < THERESHOLD:
            print(f'The file {file} has failed linting check!')
            has_failed = True

    if has_failed:
        sys.exit(1)

    print('linting succeeded!')


if __name__ == '__main__':
    lint_component()
