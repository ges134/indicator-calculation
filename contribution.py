"""
This module generates contribution graphs from PCA results.

This module is considered deprecated since the notebooks now generate the contributions graph,
which is then used in the paper. The code may still be invoked to obtain a variant in which the
indicator identifiers are present in the contribution graph.
"""

from numpy.typing import NDArray
from matplotlib.pyplot import figure, xlabel, ylabel, plot, annotate, grid, savefig, hlines, vlines
from typing import List
from tqdm import tqdm

def make_loading_plot(eigen_vectors: NDArray, codes: List[str], file_name: str):
    """
    Generates the contribution diagram based on a PCA and saves it.

    The PCA should be done beforehand with the method `stats.apply_pca`. 

    Args:
        - eigen_vectors: The eigen vectors of the resulting PCA. Only the first two PCs are
            considered for the loading graph.
        - codes: The identifiers of the indicators associated to this PCA.
        - file_name: The relative path of the saved file of the loading plot, relative to the root
            of the project.
    """
    figure(figsize=(8, 8))
    xlabel('Principal Component 1')
    ylabel('Principal Component 2')

    for i, contribution in tqdm(
        enumerate(eigen_vectors[:, :2]),
        "Tracing loading plot",
        len(eigen_vectors),
        False
    ):
        plot(contribution[0], contribution[1], 'r.')
        plot([0, contribution[0]], [0, contribution[1]], 'r-')
        annotate(codes[i], (contribution[0], contribution[1]))

    grid(True)
    savefig(file_name)

def make_loading_plot_with_confidence_intervals(
    eigen_vectors: NDArray,
    codes: List[str],
    file_name: str,
    lower_confidence_intervals: NDArray,
    upper_confidence_intervals: NDArray
):
    """
    Generates a contribution diagram based on a PCA, with confidence intervals computed using the
    Bootstrap method. The resulting graph is then saved.

    The PCA should be performed beforehand using the `stats.apply_pca` method. The bootstraped
    dataset should be generated using `confidence.bootstrap_and_apply_pca`,
    `confidence.jacknife_and_apply_pca`, and `confidence.produce_confidence_intervals`.

    Args:
        - eigen_vectors: The eigen vectors of the resulting PCA. Only the first two PCs are
            considered for the loading graph.
        - codes: The identifiers of the indicators associated with this PCA.
        - file_name: The relative path of the saved file of the loading plot, relative to the root
            of the project.
        - lower_confidence_interval: The lower bounds of the confidence intervals.
        - upper_confidence_interval: The upper bounds of the confidence intervals.
    """

    figure(figsize=(8, 8))
    xlabel('Principal Component 1')
    ylabel('Principal Component 2')

    for i, contribution in enumerate(eigen_vectors[:, :2]):
        hlines(
            contribution[1],
            lower_confidence_intervals[i][0],
            upper_confidence_intervals[i][0],
            'black'
        )
        vlines(
            contribution[0],
            lower_confidence_intervals[i][1],
            upper_confidence_intervals[i][1],
            'black'
        )
        plot(contribution[0], contribution[1], 'r.')
        plot([0, contribution[0]], [0, contribution[1]], 'r-')
        annotate(codes[i], (contribution[0], contribution[1]))

    grid(True)
    savefig(file_name)
