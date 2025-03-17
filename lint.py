"""
The scripts run the linting in all the files of the proof-of-concept program.

The script fails if the score obtained by pylint is lower than
a defined thereshold.

For each step, specific disabled rules has been added.
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
        'tests/test_merger.py',
        'tests/test_independance.py'
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
