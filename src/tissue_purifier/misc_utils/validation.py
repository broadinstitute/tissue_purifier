import torch
from typing import Callable


def create_null_distribution(
        cell_types_n: torch.Tensor,
        counts_ng: torch.Tensor,
        fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
    """
    Args:
        cell_types_n: 1D tensor or array with the cell_type information
        counts_ng: 2D array with the counts for each cell n and gene g
        fn: any function which compare the gene expression of two random cells (of the same types)
            and return a measure of similarity

    Returns:
         dict where the keys are the cell_types and the values are a list of the computed statistics for all
         possible pairs of cells
    """

    result_dict = dict()
    unique_cell_types = torch.unique(cell_types_n)
    for k, ctype in enumerate(unique_cell_types):
        counts_kg = counts_ng[cell_types_n == ctype]
        n_max = counts_kg.shape[0]
        size = int(0.5 * n_max * (n_max + 1))
        n = 0
        for i1 in range(n_max):
            for i2 in range(i1+1, n_max):
                sim = fn(counts_kg[i1, :], counts_kg[i2, :])
                if i1 == 0 and i2 == 1:
                    similarity = torch.zeros(size, sim.shape[-1])
                similarity[n, :] = sim
                n += 1
        result_dict["cell_type_"+str(k)] = similarity
    return result_dict
