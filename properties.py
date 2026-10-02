"""
This script tests the method's properties to ensure it's mathematically coherent. The script
tests 4 properties. In its current form, it validates all four properties.

The properties are as follows:

1. Two perfectly correlated criteria give an angle of 0 or 180 and a degree of independence of 0.
2. Two perfectly uncorrelated criteria give an angle of 90 and a degree of independence of 1.
3. Two criteria that are the opposite in their correlation coefficient (r2 == -r1) give the same
degree of independence.
4. Reordering of criteria should only move the angles in the matrix while preserving the values.

To test this, this script builds three matrices to test, respectively:

* properties 1 and 2, 
* property 3
* property 4

The matrix for property 4 reorders the columns of the matrix for property 3. The first column
becomes the third, the second becomes the first, and the third becomes the second.

The properties can be validated on the correlation matrix. However, since the program works with
historical data, the script will replicate data to achieve said correlation matrix. Each matrix is
tested as a 3xn matrix to ensure the correlation matrix is singular and that the third
eigenvector is equal to 0 to 14 significant digits.

The verifications that are put in place are as follows:
1. Correlation matrix
2. Eigenvalues
3. Angles between indicators
4. Degree of independence

All values are displayed on screen and reported in the manuscript.
"""

from stats import apply_pca
from numpy import array, corrcoef
from independance import get_degrees_of_independance

def main():
    """
    Main execution of the program.
    """
    # This decision matrix validates properties 1 and 2
    decision_matrix_p12 = array([
        [5, 10, 20],
        [10, 15, 5],
        [15, 20, 20],
        [20, 25, 15],
    ]).T

    # This decision matrix validates property 3.
    decision_matrix_p3 = array([
        [5, 5, -5],
        [10, 20, -20],
        [15, 15, -15],
        [20, 10, -10],
        [25, 15, -15],
        [30, 25, -25]
    ]).T

    # This decision matrix validates property 4.
    decision_matrix_p4 = array([
        [5, -5, 5],
        [20, -20, 10],
        [15, -15, 15],
        [10, -10, 20],
        [15, -15, 25],
        [25, -25, 30]
    ]).T

    print("correlation coefficients")
    print(corrcoef(decision_matrix_p12))
    print(corrcoef(decision_matrix_p3))
    print(corrcoef(decision_matrix_p4))
    print("")

    eigenvalues_p12, eigenvectors_p12, _ = apply_pca(decision_matrix_p12.T)
    eigenvalues_p3, eigenvectors_p3, _ = apply_pca(decision_matrix_p3.T)
    eigenvalues_p4, eigenvectors_p4, _ = apply_pca(decision_matrix_p4.T)

    # Validation that the third eigenvalue is equal 0 to make sure we can test properties.
    print("Eigenvalues")
    print(eigenvalues_p12)
    print(eigenvalues_p3)
    print(eigenvalues_p4)
    print("")

    # Eigenvectors to report the results
    print("Eigenvectors")
    print(eigenvectors_p12)
    print(eigenvectors_p3)
    print(eigenvectors_p4)
    print("")

    # Round to avoid floating point errors in PCA computation.
    angles_p12, degrees_p12 = get_degrees_of_independance(eigenvectors_p12.round(14))
    angles_p3, degrees_p3 = get_degrees_of_independance(eigenvectors_p3.round(14))
    angles_p4, degrees_p4 = get_degrees_of_independance(eigenvectors_p4.round(14))

    print("Angles")
    print(angles_p12)
    print(angles_p3)
    print(angles_p4)
    print("")

    print("Degrees of independence")
    print(degrees_p12)
    print(degrees_p3)
    print(degrees_p4)
    print("")

if __name__ == '__main__':
    main()
