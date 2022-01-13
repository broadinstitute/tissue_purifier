import numpy
import torch
import matplotlib
from matplotlib import pyplot as plt
from typing import Tuple, List


def _plot_single_embeddings(
        ax: "matplotlib.axes.Axes",
        embeddings: numpy.ndarray,
        colors: numpy.ndarray = None,
        labels: numpy.ndarray = None,
        x_label: str = None,
        y_label: str = None,
        title: str = None,
        size: int = 10,
        cmap_colors: "matplotlib.colors.ListedColormap" = plt.cm.inferno,
        cmap_labels: "matplotlib.colors.ListedColormap" = plt.cm.tab20):
    if colors is not None and labels is None:
        scatter_plot = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, cmap=cmap_colors, s=size)
        plt.colorbar(scatter_plot, ax=ax)
    elif colors is None and labels is not None:
        unique_labels = numpy.unique(labels)
        colors_tmp = numpy.array(cmap_labels.colors)[:, :3]
        indices = numpy.linspace(0, colors_tmp.shape[0] - 1, unique_labels.shape[0], dtype=int)
        discrete_colors = colors_tmp[indices, :]

        for n, l in enumerate(unique_labels):
            mask = (labels == l)
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1], color=discrete_colors[n], label=l, s=size)
        if len(unique_labels) < 15:
            ax.legend()
    else:
        raise Exception("Exactly one among labels and colors need to be specified.")

    if title:
        ax.set_title(title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)


def plot_embeddings(
        input_dictionary: dict,
        embedding_key: str,
        annotation_keys: List[str],
        x_label: str = None,
        y_label: str = None,
        sup_title: str = None,
        titles: List[str] = None,
        size: int = 10,
        n_col: int = 3,
        cmap_regression: "matplotlib.colors.ListedColormap" = plt.cm.inferno,
        cmap_labels: "matplotlib.colors.ListedColormap" = plt.cm.tab20,
        figsize: Tuple[float, float] = None) -> "matplotlib.pyplot.figure":

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

    def _is_color(_y):
        many_labels = len(numpy.unique(_y)) > 100
        is_float = isinstance(_y[0].item(), float)
        return many_labels * is_float

    assert isinstance(annotation_keys, list) and set(annotation_keys).issubset(set(input_dictionary.keys())), \
        "Error. Annotation_keys must be a list of keys all of which are present in the input dictionary."
    assert isinstance(embedding_key, str) and embedding_key in input_dictionary.keys(), \
        "Error. Embedding_key is not present in the input dictionary."
    assert isinstance(n_col, int) and n_col >= 1, "n_col must be an integer >= 1. Received {0}".format(n_col)
    n_max = len(annotation_keys)

    assert titles is None or (isinstance(titles, list) and len(titles) == n_max), \
        "Tiles is either None or a list of length len(annotation_keys) = {0}".format(n_max)
    assert sup_title is None or isinstance(sup_title, str), \
        "Sup_tile is either None or a string. Received {0}".format(embedding_key)

    n_col = min(n_col, n_max)
    n_row = int(numpy.ceil(float(n_max) / n_col))
    figsize = (4 * n_col, 4 * n_row) if figsize is None else figsize

    embeddings = input_dictionary[embedding_key]
    assert isinstance(embeddings, torch.Tensor) or isinstance(embeddings, numpy.ndarray), \
        "Error. Embeddings must be a torch.Tensor or a numpy.ndarray. Received {0}.".format(type(embeddings))
    embeddings_np = _preprocess_to_numpy(embeddings[:, :2])

    fig, axes = plt.subplots(ncols=n_col, nrows=n_row, figsize=figsize)
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

        if not _is_color(annotation_np):
            _ = _plot_single_embeddings(
                ax=ax_curr,
                embeddings=embeddings_np,
                x_label=x_label,
                y_label=y_label,
                title=title,
                size=size,
                colors=None,
                labels=annotation_np,
                cmap_colors=None,
                cmap_labels=cmap_labels)
        else:
            _ = _plot_single_embeddings(
                ax=ax_curr,
                embeddings=embeddings_np,
                x_label=x_label,
                y_label=y_label,
                title=title,
                size=size,
                colors=annotation_np,
                labels=None,
                cmap_colors=cmap_regression,
                cmap_labels=None)

    if sup_title:
        fig.suptitle(sup_title)
    fig.tight_layout()
    plt.close(fig)
    return fig


### OLD BUT GOOD STUFF
##def plot_embeddings(
##        embeddings: Union[numpy.array, torch.Tensor],
##        x_label: Optional[str] = None,
##        y_label: Optional[str] = None,
##        title: Optional[str] = None,
##        size: int = 10,
##        cmap: "matplotlib.colors.ListedColormap" = plt.cm.inferno,
##        colors: Optional[numpy.array] = None,
##        labels: Optional[list] = None,
##        figsize: Optional[Tuple[float, float]] = None,
##        legend: bool = True,
##        ):
##
##    assert len(embeddings.shape) == 2 and embeddings.shape[-1] == 2  # i.e. embeddings.shape = (*, 2)
##
##    assert ((colors is None) and (labels is not None)) or ((colors is not None) and (labels is None)), (
##        "One and only one between labels and colors need to be defined"
##    )
##
##    def _preprocess_to_numpy(_y) -> numpy.ndarray:
##        if isinstance(_y, torch.Tensor):
##            return _y.cpu().detach().numpy()
##        elif isinstance(_y, list):
##            return numpy.array(_y)
##        elif isinstance(_y, numpy.ndarray):
##            return _y
##        else:
##            raise Exception(
##                "Labels is either None or torch.Tensor, List, numpy.array. Received {0}".format(type(_y)))
##
##    fig = plt.figure(figsize=figsize)
##
##    embeddings = _preprocess_to_numpy(embeddings)
##
##    if colors is not None:
##        np_colors = _preprocess_to_numpy(colors),
##        plt.scatter(embeddings[:, 0], embeddings[:, 1], c=np_colors, cmap=cmap, s=size)
##        plt.colorbar()
##    elif labels is not None:
##        np_labels = _preprocess_to_numpy(labels)
##        unique_labels = numpy.unique(np_labels)
##        colors = numpy.array(cmap.colors)[:, :3]
##        indices = numpy.linspace(0, colors.shape[0] - 1, unique_labels.shape[0], dtype=int)
##        discrete_colors = colors[indices, :]
##
##        for n, l in enumerate(unique_labels):
##            mask = (np_labels == l)
##            plt.scatter(embeddings[mask, 0], embeddings[mask, 1], color=discrete_colors[n], label=l, s=size)
##
##        if legend:
##            plt.legend()
##
##    if x_label is not None:
##        plt.xlabel(x_label)
##    if y_label is not None:
##        plt.ylabel(y_label)
##    if title is not None:
##        plt.title(title)
##    plt.close(fig)
##    return fig
##
##
##def plot_all_maps(
##        input_dict: dict,
##        cmap: "matplotlib.colors.ListedColormap" = plt.cm.inferno,
##        figsize: Optional[Tuple[float, float]] = None,
##        legend: bool = True,) -> List:
##    """
##    Assumes that the input_dictionary has keys starting with:
##    umap_, pca_, regress_, classify_, leiden_
##    """
##
##    maps = []
##
##    def _preprocess(v):
##        if isinstance(v, list):
##            return v
##        elif isinstance(v, torch.Tensor):
##            return v.cpu().numpy()
##        elif isinstance(v, numpy.ndarray):
##            return v
##        else:
##            raise Exception("Expected list, torch.Tensor, numpy.ndarray but received {0}".format(type(v)))
##
##    # Separate keys in 5 disjoint sets
##    k_color, k_label, k_umap, k_pca, k_leiden = [], [], [], [], []
##    for k in input_dict.keys():
##        if k.startswith("umap"):
##            k_umap.append(k)
##        elif k.startswith("pca"):
##            k_pca.append(k)
##        elif k.startswith("classify"):
##            k_label.append(k)
##        elif k.startswith("leiden"):
##            k_leiden.append(k)
##        elif k.startswith("regress"):
##            k_color.append(k)
##
##    # create nested list to plot UMAP
##    features, labels, colors, titles = [], [], [], []
##    for key in k_umap:
##        for kc in k_color:
##            titles.append(key+" by "+kc)
##            features.append(_preprocess(input_dict[key]))
##            labels.append(None)
##            colors.append(_preprocess(input_dict[kc]))
##        for kl in k_label:
##            titles.append(key + " by " + kl)
##            features.append(_preprocess(input_dict[key]))
##            labels.append(_preprocess(input_dict[kl]))
##            colors.append(None)
##        for k_cluster in k_leiden:
##            # Make sure that I am plotting the Leiden colors which were computed using the same features
##            if k_cluster.endswith(key.lstrip("umap_")):
##                titles.append(k_cluster)
##                features.append(_preprocess(input_dict[key]))
##                labels.append(numpy.argmax(_preprocess(input_dict[k_cluster]), axis=-1))  # this undo the one-hot operation
##                colors.append(None)
##
##    # finally plot UMAP
##    for f, l, c, t in zip(features, labels, colors, titles):
##        tmp = plot_embeddings(
##            embeddings=f[:, :2],
##            labels=l,
##            colors=c,
##            x_label="UMAP 1",
##            y_label="UMAP 2",
##            title=t,
##            cmap=cmap,
##            figsize=figsize,
##            legend=legend,
##        )
##        maps.append(tmp)
##
##    # create nested list to plot PCA
##    features, labels, colors, titles = [], [], [], []
##    for key in k_pca:
##        for kc in k_color:
##            titles.append(key + " by " + kc)
##            features.append(_preprocess(input_dict[key]))
##            labels.append(None)
##            colors.append(_preprocess(input_dict[kc]))
##        for kl in k_label:
##            titles.append(key + " by " + kl)
##            features.append(_preprocess(input_dict[key]))
##            labels.append(_preprocess(input_dict[kl]))
##            colors.append(None)
##        for k_cluster in k_leiden:
##            # Make sure that I am plotting the Leiden colors which were computed using the same features
##            if k_cluster.endswith(key.lstrip("pca_")):
##                titles.append(k_cluster)
##                features.append(_preprocess(input_dict[key]))
##                labels.append(numpy.argmax(_preprocess(input_dict[k_cluster]), axis=-1))  # this undo the one-hot operation
##                colors.append(None)
##
##    # finally plot PCA
##    for f, l, c, t in zip(features, labels, colors, titles):
##        tmp = plot_embeddings(
##            embeddings=f[:, :2],
##            labels=l,
##            colors=c,
##            x_label="PCA 1",
##            y_label="PCA 2",
##            title=t,
##            cmap=cmap,
##            figsize=figsize,
##            legend=legend,
##        )
##        maps.append(tmp)
##
##    return maps
##