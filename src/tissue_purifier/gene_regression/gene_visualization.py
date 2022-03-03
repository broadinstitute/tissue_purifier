import torch
import matplotlib.pyplot as plt
import numpy


def plot_few_gene_hist(cell_types_n, value1_ng, value2_ng=None, bins=20):
    """
    Plot the per cell-type histogram.

    Args:
        cell_types_n: tensor of shape N with the cell type information
        value1_ng: first quantity to plot of shape (N,G)
        value2_ng: the secont quantity to plot of shape (N,G)
        bins: number of bins in the histogram

    Returns:
        A figure with G rows and K columns where K is the number of distinct cell types
    """

    assert len(cell_types_n.shape) == 1
    assert len(value1_ng.shape) >= 2
    assert cell_types_n.shape[0] == value1_ng.shape[-2]
    assert value2_ng is None or (value1_ng.shape == value2_ng.shape)

    def _to_torch(_x):
        if isinstance(_x, torch.Tensor):
            return _x
        elif isinstance(_x, numpy.ndarray):
            return torch.tensor(_x)
        else:
            raise Exception("Expected torch.tensor or numpy.ndarray. Received {0}".format(type(_x)))

    value2_ng = None if value2_ng is None else _to_torch(value2_ng)
    value1_ng = _to_torch(value1_ng)
    ctypes = torch.unique(cell_types_n)
    genes = value1_ng.shape[-1]

    nrows = genes
    ncols = len(ctypes)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(4 * ncols, 4 * nrows))

    for r in range(genes):
        tmp = value1_ng[..., r]
        other_tmp = None if value2_ng is None else value2_ng[..., r]
        for c, c_type in enumerate(ctypes):
            if r == 0:
                _ = axes[r, c].set_title("cell_type {0}".format(c_type))

            if value2_ng is None:
                tmp2 = tmp[..., cell_types_n == c_type]

                if tmp2.dtype == torch.long:
                    y = torch.bincount(tmp2)
                    x = torch.arange(y.shape[0]+1)
                else:
                    y, x = numpy.histogram(tmp2, bins=bins, density=True)
                barWidth = 0.9 * (x[1] - x[0])
                _ = axes[r, c].bar(x[:-1], y, width=barWidth)
            else:
                tmp2 = tmp[..., cell_types_n == c_type].flatten()
                other_tmp2 = other_tmp[..., cell_types_n == c_type].flatten()
                myrange = (min(min(tmp2), min(other_tmp2)).item(), max(max(tmp2), max(other_tmp2)).item())

                if tmp2.dtype == torch.long:
                    y = torch.bincount(tmp2, minlength=myrange[1])
                    other_y = torch.bincount(other_tmp2, minlength=myrange[1])
                    x = torch.arange(myrange[1]+1)
                    other_x = torch.arange(myrange[1]+1)
                else:
                    y, x = numpy.histogram(tmp2, range=myrange, bins=bins, density=True)
                    other_y, other_x = numpy.histogram(other_tmp2, range=myrange, bins=bins, density=True)

                barWidth = 0.4 * (x[1] - x[0])
                _ = axes[r, c].bar(x[:-1], y, width=barWidth)
                _ = axes[r, c].bar(other_x[:-1] + barWidth, other_y, width=barWidth)

    plt.close()
    return fig

