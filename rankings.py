"""
This script computes Kendall's tau with the given rankings. Rankings are copied from the results.
The program prints the values, which are then reported back in the manuscript.
"""

from scipy.stats import kendalltau

def main():
    """
    Main execution of the program.
    """
    integrated = [
        6,
        18,
        16,
        8,
        1,
        14,
        15,
        7,
        5,
        25,
        4,
        23,
        22,
        19,
        3,
        24,
        17,
        2,
        9,
        10,
        21,
        26,
        13,
        20,
        11,
        12
    ]

    original = [
        6,
        18,
        17,
        10,
        1,
        13,
        15,
        7,
        4,
        26,
        3,
        25,
        22,
        20,
        5,
        23,
        16,
        2,
        9,
        8,
        21,
        24,
        14,
        19,
        11,
        12
    ]

    res = kendalltau(original, integrated)
    print(res.statistic)

if __name__ == '__main__':
    main()
