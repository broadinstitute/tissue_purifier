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


def plot_multiple_barplots(data: "panda.DataFrame",
                           x: str,
                           ys: List[str],
                           n_col: int = 4,
                           figsize: Tuple[float, float] = None,
                           y_labels: List[str] = None,
                           x_labels_rotation: int = 90,
                           x_labels: List[str] = None,
                           titles: List[str] = None,
                           y_lims: Tuple[float, float] = None,
                           **kargs):
    """ Takes a dataframe and make multiple bar plots

    Args:
        kargs: any argument passed to seaborn.barplot such as hue,

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


### OLD BUT GOOD STUFF
##def plot_bars(
##        y_values: Any,
##        x_labels: List[str] = None,
##        figsize: Tuple[int, int] = (9, 9),
##        title: str = None,):
##
##    assert x_labels is None or (len(x_labels) == len(y_values))
##
##    X_axis = torch.arange(len(y_values)).numpy()
##
##    fig, ax = plt.subplots(1, 1, figsize=figsize)
##    _ = ax.bar(X_axis, y_values, width=0.9)
##
##    if x_labels:
##        ax.set_xticks(X_axis)
##        ax.set_xticklabels(x_labels, rotation=90)
##    else:
##        ax.set_xticks(X_axis)
##
##    if title:
##        fig.suptitle(title)
##    fig.tight_layout()
##    plt.close(fig)
##    return fig
##
##
##def plot_composition(
##        composition: Union[torch.Tensor, List[torch.Tensor]],
##        x_labels: List[str] = None,
##        dataset_labels: List[str] = None,
##        figsize: Tuple[int, int] = (9, 9),
##        title: str = None,
##):
##
##    if isinstance(composition, torch.Tensor):
##        composition = [composition]
##    else:
##        for i in range(len(composition) - 1):
##            if composition[i].shape != composition[i + 1].shape:
##                raise Exception("counters have different shapes. Can not be plotted together")
##
##    assert dataset_labels is None or len(dataset_labels) == len(composition)
##
##    X_axis = torch.arange(composition[0].shape[0]).numpy()
##
##    n_datasets = len(composition)
##    width = 0.9 / n_datasets
##    fig, ax = plt.subplots(1, 1, figsize=figsize)
##    for n, counter in enumerate(composition):
##        label = dataset_labels[n] if dataset_labels else None
##        _ = ax.bar(X_axis + n * width, counter.cpu().numpy(), width, label=label)
##
##    if dataset_labels:
##        ax.legend()
##
##    if x_labels:
##        ax.set_xticks(X_axis)
##        ax.set_xticklabels(x_labels, rotation=90)
##    else:
##        ax.set_xticks(X_axis)
##
##    if title:
##        fig.suptitle(title)
##    fig.tight_layout()
##    plt.close(fig)
##    return fig
##def plot_predictions_as_bars(input_dict, feature_keys, annotation_keys, y_lim: Tuple[float, float],
##                             figsize: Tuple[int, int] = None, sup_title: str = None):
##    n_annotations = len(annotation_keys)
##    figsize = (4 * n_annotations, 4) if figsize is None else figsize
##    fig, axes = plt.subplots(ncols=n_annotations, figsize=figsize)
##
##    for n, annotation_key in enumerate(annotation_keys):
##        keys_tmp, y_means, y_stds = [], [], []
##        for k in input_dict.keys():
##            if annotation_key in k:
##                keys_tmp.append(k)
##                y_value_tmp = input_dict[k]
##                if len(y_value_tmp) == 2:
##                    y_means.append(y_value_tmp[0])
##                    y_stds.append(y_value_tmp[1])
##                else:
##                    y_means.append(y_value_tmp)
##                    y_stds.append(None)
##
##        labels_tmp = []
##        for k in keys_tmp:
##            for kf in feature_keys:
##                if kf in k:
##                    labels_tmp.append(kf)
##
##        _plot_bars(ax=axes[n], y_means=y_means, y_stds=y_stds, x_labels=labels_tmp, y_lim=y_lim, title=annotation_key)
##
##    if sup_title:
##        fig.suptitle(sup_title)
##
##    fig.tight_layout()
##    plt.close(fig)
##    return fig
##
##