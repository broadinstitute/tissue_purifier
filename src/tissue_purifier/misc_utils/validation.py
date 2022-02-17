import torch


def _make_binning(_x_ng, _boundaries):
    """ Bins the metric for each cell and gene given the boundaries tensor. """
    b = _boundaries.shape[0]  # shape: (bins)
    n, g = _x_ng.shape[:2]   # shape: (n_sample, n_genes)

    _index_ng = torch.bucketize(_x_ng, _boundaries)
    _src = torch.ones((n, g, b+1))
    _counter = torch.zeros_like(_src).scatter(dim=-1, index=_index_ng.unsqueeze(dim=-1), src=_src)
    return _counter.sum(dim=0)  # sum over sample. shape: (genes, bins+1)


def create_null_distribution(
        cell_types_n: torch.Tensor,
        counts_ng: torch.Tensor,
        similarity_measure: str = "L1",
        boundaries: torch.Tensor = torch.linspace(start=0, end=30, steps=31)) -> dict:
    """
    Compute the similarity between all possible pairs of cells cells of the same type.
    The similarity is binned and the resulting histogram is returned.

    Args:
        cell_types_n: 1D tensor or array with the cell_type information
        counts_ng: 2D array with the counts for each cell n and gene g
        similarity_measure: type of statistic to compute between gene expressions. If defaults to the L1
        boundaries: this specify how to bin the data

    Returns:
         null_dict. Dictionary with distribution of the statistis under the null model.
            The dictionary includes the bins used and the bin_counter for each gene.
    """

    if torch.cuda.is_available():
        cell_types_n = cell_types_n.cuda()
        counts_ng = counts_ng.cuda()
        boundaries = boundaries.cuda()

    result_dict = {"boundaries": boundaries}
    unique_cell_types = torch.unique(cell_types_n)
    g = counts_ng.shape[-1]

    for k, ctype in enumerate(unique_cell_types):
        counts_kg = counts_ng[cell_types_n == ctype]

        bin_counter_gb = torch.zeros(g, boundaries.shape[0]+1).cpu().numpy()
        for i1 in range(1, counts_kg.shape[0]):
            gene_exp_others_mg = counts_kg[:i1, :]
            gene_exp_ref_1g = counts_kg[i1, :]

            if similarity_measure == "L1":
                sim_mg = (gene_exp_ref_1g - gene_exp_others_mg).abs()
            else:
                raise ValueError("similarity_measure not valid. Received {0}".format(similarity_measure))

            bin_increment_gb = _make_binning(sim_mg, boundaries)
            bin_counter_gb += bin_increment_gb.cpu().numpy()

        result_dict["cell_type_"+str(k)] = bin_counter_gb[..., :-1]
    return result_dict


def create_heldout_distribution(
        cell_types_n: torch.Tensor,
        true_counts_ng: torch.Tensor,
        pred_counts_ng: torch.Tensor,
        similarity_measure: str = "L1",
        boundaries: torch.Tensor = torch.linspace(start=0, end=30, steps=31)) -> dict:
    """
    Compute the similarity between all possible pairs of cells cells of the same type.
    The similarity is binned and the resulting histogram is returned.

    Args:
        cell_types_n: 1D tensor or array with the cell_type information
        true_counts_ng: 2D array with the ture counts for each cell n and gene g
        pred_counts_ng: 2D array with the predicted counts for each cell n and gene g
        similarity_measure: type of statistic to compute between gene expressions. If defaults to the L1
        boundaries: this specify how to bin the data

    Returns:
         heldout_dict. Dictionary with distribution of the statistics for the heldout data.
            The dictionary includes the bins used and the bin_counter for each gene.
    """

    if torch.cuda.is_available():
        cell_types_n = cell_types_n.cuda()
        true_counts_ng = true_counts_ng.cuda()
        pred_counts_ng = pred_counts_ng.cuda()
        boundaries = boundaries.cuda()

    assert true_counts_ng.shape == pred_counts_ng.shape
    assert cell_types_n.shape[0] == true_counts_ng.shape[0]

    result_dict = {"boundaries": boundaries}
    unique_cell_types = torch.unique(cell_types_n)
    g = true_counts_ng.shape[-1]

    for k, ctype in enumerate(unique_cell_types):
        mask_k_type = (cell_types_n == ctype)
        true_counts_kg = true_counts_ng[mask_k_type]
        pred_counts_kg = pred_counts_ng[mask_k_type]

        bin_counter_gb = torch.zeros(g, boundaries.shape[0]+1).cpu().numpy()

        if similarity_measure == "L1":
            sim_mg = (true_counts_kg - pred_counts_kg).abs()
        else:
            raise ValueError("similarity_measure not valid. Received {0}".format(similarity_measure))

        bin_increment_gb = _make_binning(sim_mg, boundaries)
        bin_counter_gb += bin_increment_gb.cpu().numpy()

        result_dict["cell_type_"+str(k)] = bin_counter_gb[..., :-1]
    return result_dict
