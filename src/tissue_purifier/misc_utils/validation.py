import torch


def create_null_distribution(
        cell_types_n: torch.Tensor,
        counts_ng: torch.Tensor,
        similarity_measure: str = "L1",
        boundaries: torch.Tensor = torch.linspace(start=0, end=100, steps=101),
        interval_close_to_the_right: bool = False) -> dict:
    """
    Create the null distribution by comparing all possible pair of cells of the same type.
    This can be very computationally end memory expensive.
    Consider passing in a subset of cell_types and/or genes.

    Args:
        cell_types_n: 1D tensor or array with the cell_type information
        counts_ng: 2D array with the counts for each cell n and gene g
        similarity_measure: type of statistic to compute between gene expressions. If defaults to the L1
        boundaries: the statistics will be binned in these intervals. Must be in increasing order.
        interval_close_to_the_right: If False (Defaults) returns index i
            which satisfies boundaries[i-1] < x <= boundaries[i]
            If True, returns i such that boundaries[i-1] <= x < boundaries[i]

    Note:
        Values are clipped to edges, i.e. if x > boundaries[-1] it return the last index.



    Returns:
         null_dict. Dictionary with distribution of the statistis under the null model.
            The dictionary includes the bins used and the bin_counter for each gene.
    """

    def _make_binning(_x_ng, _bins):
        mask_bng = (_x_ng < _bins[:, None, None])
        # find the last true
        index_bng =
        _x_ng <

    result_dict = {"bins": bins}
    unique_cell_types = torch.unique(cell_types_n)
    g = counts_ng.shape[-1]
    n_bins = bins.shape[0]

    for k, ctype in enumerate(unique_cell_types):
        counts_kg = counts_ng[cell_types_n == ctype]

        bin_counter = torch.zeros(g, n_bins).cpu().numpy()
        for i1 in range(1, counts_kg.shape[0]):
            gene_exp_others = counts_kg[:i1, :]
            gene_exp_ref = counts_kg[i1, :]

            if similarity_measure == "L1":
                sim = (gene_exp_ref - gene_exp_others).abs()
            else:
                raise ValueError("similarity_measure not valid. Received {0}".format(similarity_measure))

            index_ng = torch.bucketize(sim, boundaries, right=interval_close_to_the_right)
            torch.bincount()


            assert bin_increment.shape == torch.Size([g, n_bins])
            bin_counter += bin_increment.cpu().numpy()

        result_dict["cell_type_"+str(k)] = bin_counter
    result_dict["bins"] = bins
    return result_dict
