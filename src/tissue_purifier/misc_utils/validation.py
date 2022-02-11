import torch
from typing import Callable


def create_null_distribution(
        cell_types_n: torch.Tensor,
        counts_ng: torch.Tensor,
        similarity_measure: str = "L1"):
    """
    Args:
        cell_types_n: 1D tensor or array with the cell_type information
        counts_ng: 2D array with the counts for each cell n and gene g
        similarity_measure: type of statistic to compute between gene expressions. If defaults to the L1

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

        for i1 in range(n_max):
            mask = (torch.arange(n_max) == i1)
            gene_exp_ref = counts_kg[mask, :]
            gene_exp_others = counts_kg[~mask, :]
            if similarity_measure == "L1":
                sim = (gene_exp_ref - gene_exp_others).abs().cpu().numpy()
            else:
                raise ValueError("similarity_measure not valid. Received {0}".format(similarity_measure))

            if i1 == 0:
                similarity = torch.zeros(size, sim.shape[-1]).cpu().numpy()

            i_start = i1*(n_max-1)
            i_end = (i1+1)*(n_max-1)
            print(i_start, i_end)
            similarity[i_start:i_end, :] = sim

        result_dict["cell_type_"+str(k)] = similarity
    return result_dict
