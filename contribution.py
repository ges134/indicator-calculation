"""
This modules generated a contribution plot based on the PCA results.
"""

from numpy.typing import NDArray
from matplotlib.pyplot import figure, xlabel, ylabel, plot, annotate, grid, savefig
from typing import List
from tqdm import tqdm

def make_loading_plot(eigen_vectors: NDArray, codes: List[str], file_name: str):
    """
    Generates the loading plot based on a PCA and saves it.

    The PCA should be done beforehand with the method `stats.apply_pca`. 

    Args:
        - eigen_vectors: The eigen vectors of the resulting PCA. Only the firt two PCs are
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
