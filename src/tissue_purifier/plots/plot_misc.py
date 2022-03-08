import matplotlib
from matplotlib import pyplot as plt
from typing import Tuple, Any, List, Union
import numpy
import torch
import pandas
import seaborn


def plot_cdf_pdf(
        cdf_y: Union[numpy.ndarray, torch.Tensor] = None,
        pdf_y: Union[numpy.ndarray, torch.Tensor] = None,
        x_label: str = None,
        sup_title: str = None) -> "matplotlib.pyplot.figure":

    if cdf_y is not None and pdf_y is None:
        pdf_y = cdf_y.clone()
        for i in range(1, len(cdf_y)):
            pdf_y[i] = cdf_y[i] - cdf_y[i - 1]
        pdf_y[0] = cdf_y[0]
    elif cdf_y is None and pdf_y is not None:
        cdf_y = numpy.cumsum(pdf_y, axis=0)

    fig, axes = plt.subplots(ncols=2, figsize=(4 * 2, 4))
    _ = axes[0].plot(pdf_y, '.')
    _ = axes[0].set_ylabel("pdf")

    _ = axes[1].plot(cdf_y, '.')
    _ = axes[1].set_ylabel("cdf")
    if x_label:
        _ = axes[0].set_xlabel(x_label)
        _ = axes[1].set_xlabel(x_label)

    if sup_title:
        fig.suptitle(sup_title)

    # fig.tight_layout()
    plt.close(fig)
    return fig


def _plot_multigroup_bars(
        ax: "matplotlib.axes.Axes",
        y_values: Union[torch.Tensor, numpy.ndarray],
        y_errors: Union[torch.Tensor, numpy.ndarray] = None,
        x_labels: List[Any] = None,
        group_labels: List[Any] = None,
        title: str = None,
        group_legend: bool = None,
        y_lim: Tuple[float, float] = None,
        ):
    """
    Make a bar plot of a tensor of shape (groups, x_locs).
    Each x_loc will have n_groups bars shown next to each other.

    Args:
        ax: the current axes to draw the the bars
        y_values: tensor of shape: (groups, x_locs) with the means
        y_errors: tensor of shape: (groups, x_locs) with the stds (optional)
        x_labels: List[str] of length N_types
        group_labels: List[str] of length N_groups
        title: string. The title of the plot
        group_legend: bool. If true show the group legend.
        y_lim: Tuple[float, float] specifies the extension of the y_axis. For example y_lim = (0.0, 1.0)
    """

    assert y_errors is None or y_errors.shape == y_values.shape

    if len(y_values.shape) == 1:
        n_groups = 1
        n_values = y_values.shape[0]

        # add singleton dimension
        y_values = y_values[None, :]
        y_errors = None if y_errors is None else y_errors[None, :]

    elif len(y_values.shape) == 2:
        n_groups, n_values = y_values.shape
    else:
        raise Exception("y_values must be a 1D or 2D array (if multiple groups). Received {0}.".format(y_values.shape))

    assert x_labels is None or (isinstance(x_labels, list) and len(x_labels) == n_values)
    assert group_labels is None or (isinstance(group_labels, list) and len(group_labels) == n_groups)

    X_axis = numpy.arange(n_values)
    width = 0.9 / n_groups
    for n in range(n_groups):
        group_label = None if group_labels is None else group_labels[n]
        _ = ax.bar(X_axis + n * width, y_values[n], width, label=group_label)
        if y_errors:
            _ = ax.errorbar(X_axis + n * width, y_values[n], yerr=y_errors[n], fmt="o", color="r")

    show_legend = (group_legend is None and group_labels is not None) or group_legend
    if show_legend:
        ax.legend()

    if x_labels:
        ax.set_xticks(X_axis + 0.45)
        ax.set_xticklabels(x_labels, rotation=90)
    else:
        ax.set_xticks(X_axis + 0.45)

    if y_lim:
        ax.set_ylim(y_lim)

    if title:
        ax.set_title(title)


def plot_clusters_annotations(
        input_dictionary: dict,
        cluster_key: str,
        annotation_keys: List[str],
        titles: List[str] = None,
        sup_title: str = None,
        n_col: int = 3,
        figsize: Tuple[float, float] = None,
        ) -> "matplotlib.pyplot.figure":

    def _preprocess_to_numpy(_y) -> numpy.ndarray:
        if isinstance(_y, torch.Tensor):
            return _y.cpu().detach().numpy()
        elif isinstance(_y, list):
            return numpy.array(_y)
        elif isinstance(_y, numpy.ndarray):
            return _y
        else:
            raise Exception(
                "Labels is either None or torch.Tensor, List, numpy.array. Received {0}".format(type(_y)))

    def _is_continuous(_y) -> bool:
        is_float = isinstance(_y[0].item(), float)
        lot_of_values = len(numpy.unique(_y)) > 20
        return is_float * lot_of_values

    assert isinstance(n_col, int) and n_col >= 1, "n_col must be an integer >= 1. Received {0}".format(n_col)
    assert isinstance(annotation_keys, list) and set(annotation_keys).issubset(set(input_dictionary.keys())), \
        "Error. Annotation_keys must be a list of keys all of which are present in the input dictionary."
    assert isinstance(cluster_key, str) and cluster_key in input_dictionary.keys(), \
        "Error. Cluster_key is not present in the input dictionary."
    assert titles is None or (isinstance(titles, list) and len(titles) == len(annotation_keys)), \
        "Tiles is either None or a list of length len(annotation_keys) = {0}".format(len(annotation_keys))
    assert sup_title is None or isinstance(sup_title, str), \
        "Sup_tile is either None or a string. Received {0}".format(sup_title)

    n_max = len(annotation_keys)
    n_col = min(n_col, n_max)
    n_row = int(numpy.ceil(float(n_max) / n_col))
    figsize = (4 * n_col, 4 * n_row) if figsize is None else figsize
    fig, axes = plt.subplots(ncols=n_col, nrows=n_row, figsize=figsize)

    cluster_labels_np = _preprocess_to_numpy(input_dictionary[cluster_key])
    unique_cluster_labels = numpy.unique(cluster_labels_np)

    for n, annotation_k in enumerate(annotation_keys):

        title = None if titles is None else titles[n]

        if n_col == 1 and n_row == 1:
            ax_curr = axes
        elif n_row == 1:
            ax_curr = axes[n]
        else:
            c = n % n_col
            r = n // n_col
            ax_curr = axes[r, c]

        annotation_tmp = input_dictionary[annotation_k]
        annotation_np = _preprocess_to_numpy(annotation_tmp)

        if _is_continuous(annotation_np):
            # make violin plots
            df_tmp = pandas.DataFrame.from_dict({'clusters': cluster_labels_np, annotation_k: annotation_np})
            _ = seaborn.violinplot(x='clusters', y=annotation_k, data=df_tmp, ax=ax_curr)
        else:
            # make a multi bar-chart. I need counts of shape (n_clusters, n_unique_annotations)
            unique_annotations = numpy.unique(annotation_np)  # shape: na
            counts = numpy.zeros((len(unique_cluster_labels), len(unique_annotations)), dtype=int)
            for n1, l_cluster in enumerate(unique_cluster_labels):
                mask_cluster = (cluster_labels_np == l_cluster)
                for n2, l_annotation in enumerate(unique_annotations):
                    mask_annotation = (annotation_np == l_annotation)
                    counts[n1, n2] = (mask_cluster * mask_annotation).sum()
            _ = _plot_multigroup_bars(ax=ax_curr,
                                      y_values=counts,
                                      x_labels=unique_annotations.tolist(),
                                      group_labels=unique_cluster_labels.tolist(),
                                      group_legend=False)

        ax_curr.set_title(title)

    if sup_title:
        fig.suptitle(sup_title)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_multiple_barplots(
        data: "pandas.DataFrame",
        x: str,
        ys: List[str],
        n_col: int = 4,
        figsize: Tuple[float, float] = None,
        y_labels: List[str] = None,
        x_labels_rotation: int = 90,
        x_labels: List[str] = None,
        titles: List[str] = None,
        y_lims: Tuple[float, float] = None,
        **kargs) -> "matplotlig.figure":
    """ Takes a dataframe and make multiple bar plot leveraging
        seaborn.barplot(y=y, x=x, data=data)

    Args:
        data: a dataframe with the data to plot
        x: names of variables in data
        ys: names of variables in data
        n_col: number of columns panels in the figure
        figsize: size of the output figure
        x_labels: labels
        y_labels: labels
        x_labels_rotation: rotation in degree of the x_labels (default 90)
        titles: titles for each panel
        y_lims: set y_limits for all the panels
        kargs: any argument passed to seaborn.barplot such as hue,

    Returns:
        figure
    """

    n_max = len(ys)
    n_col = min(n_col, n_max)
    n_row = int(numpy.ceil(float(n_max) / n_col))
    figsize = (4 * n_col, 4 * n_row) if figsize is None else figsize
    fig, axes = plt.subplots(ncols=n_col, nrows=n_row, figsize=figsize)

    if titles:
        assert len(titles) == n_max
    if y_labels:
        assert len(y_labels) == n_max

    for n, y in enumerate(ys):
        if n_col == 1 and n_row == 1:
            ax_curr = axes
        elif n_row == 1:
            ax_curr = axes[n]
        else:
            c = n % n_col
            r = n // n_col
            ax_curr = axes[r, c]

        _ = seaborn.barplot(y=y, x=x, data=data, ax=ax_curr, **kargs)

        # y_lims
        if y_lims:
            ax_curr.set_ylim(y_lims[0], y_lims[1])

        # x_labels :
        x_labels_raw = ax_curr.get_xticklabels()
        if x_labels:
            assert len(x_labels) == len(x_labels_raw)
        else:
            x_labels = x_labels_raw
        ax_curr.set_xticklabels(labels=x_labels, rotation=x_labels_rotation)

        # titles
        title = ax_curr.get_ylabel() if titles is None else titles[n]
        ax_curr.set_title(title)
        ax_curr.set_xlabel(None)

        # y_labels
        if y_labels:
            ax_curr.set_ylabel(y_labels[n])
        else:
            ax_curr.set_ylabel(None)

    fig.tight_layout()
    plt.close(fig)
    return fig


def show_corr_matrix(data: torch.Tensor, show_colorbar: bool = True, sup_title: str = None):
    data = data.detach().cpu().clone()
    mask = torch.eye(data.shape[0]).bool()

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))
    _ = seaborn.heatmap(
        data=data,
        square=True,
        xticklabels=False,
        yticklabels=False,
        center=0.0,
        robust=False,  # so that I can see the full scale of diagonal and off-diagonal
        cbar=show_colorbar,
        ax=axes[0, 0])
    _ = axes[0, 0].set_title("Diagonal and off-diagonal")

    diagonal = data[mask].flatten().numpy()
    _ = seaborn.histplot(x=diagonal, kde=True, bins=100, ax=axes[1, 0])
    _ = axes[1, 0].set_title("Histogram of the diagonal element")

    data_overwritten = data.clone()
    data_overwritten[mask] = 0.0
    seaborn.heatmap(data=data_overwritten,
                    square=True,
                    xticklabels=False,
                    yticklabels=False,
                    center=0.0,
                    robust=True,
                    cbar=show_colorbar,
                    ax=axes[0, 1])
    _ = axes[0, 1].set_title("Off-diagonal only")

    off_diagonal = data[~mask].flatten().numpy()
    _ = seaborn.histplot(x=off_diagonal, kde=True, bins=100, ax=axes[1, 1])
    _ = axes[1, 1].set_title("Histogram of the off-diagonal element")

    if sup_title:
        _ = fig.suptitle(sup_title)
    plt.close(fig)
    return fig
