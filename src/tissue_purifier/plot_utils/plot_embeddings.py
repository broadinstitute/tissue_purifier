from anndata import AnnData
import pandas as pd
from scanpy.plotting import embedding
import numpy
import torch
from matplotlib.colors import Colormap
from typing import List


def plot_embeddings(
        input_dictionary: dict,
        embedding_key: str,
        annotation_keys: List[str],
        sup_title: str = None,
        n_col: int = 3,
        cmap: Colormap = None) -> "matplotlib.pyplot.figure":

    def _is_categorical(_x):
        is_float = (_x.dtype == numpy.single or _x.dtype == numpy.float or _x.dtype == numpy.double)
        is_many = (_x.shape[0] > 100)
        is_continuous = is_many and is_float
        return ~is_continuous

    # make a copy of the dict with the torch to numpy conversion
    cloned_dict = {}
    for k, v in input_dictionary.items():
        if isinstance(v, torch.Tensor):
            cloned_dict[k] = v.detach().cpu().numpy()
        else:
            cloned_dict[k] = v

    # create dataframe with annotations
    df = pd.DataFrame(cloned_dict, columns=annotation_keys)
    for k in annotation_keys:
        vec = numpy.unique(df[k].to_numpy())
        if _is_categorical(vec):
            df[k] = df[k].astype("category")

    # create anndata with annotations and embeddings
    adata = AnnData(obs=df)
    adata.obsm[embedding_key] = input_dictionary[embedding_key]

    # leverage anndata embedding capabilities
    fig = embedding(adata,
                    basis=embedding_key,
                    color=annotation_keys,
                    return_fig=True,
                    show=False,
                    ncols=n_col,
                    cmap=cmap)
    if sup_title:
        _ = fig.suptitle(sup_title)
    return fig


# def _plot_single_embeddings(
#         ax: "matplotlib.axes.Axes",
#         embeddings: numpy.ndarray,
#         colors: numpy.ndarray = None,
#         labels: numpy.ndarray = None,
#         x_label: str = None,
#         y_label: str = None,
#         title: str = None,
#         size: int = 10,
#         cmap_colors: "matplotlib.colors.ListedColormap" = plt.cm.inferno,
#         cmap_labels: "matplotlib.colors.ListedColormap" = plt.cm.tab20):
#     if colors is not None and labels is None:
#         _ = ax.set_xlim(emit=False, auto=True)
#         _ = ax.set_ylim(emit=False, auto=True)
#         scatter_plot = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, cmap=cmap_colors, s=size)
#         _ = plt.colorbar(scatter_plot, ax=ax)
#     elif colors is None and labels is not None:
#         unique_labels = numpy.unique(labels)
#         colors_tmp = numpy.array(cmap_labels.colors)[:, :3]
#         indices = numpy.linspace(0, colors_tmp.shape[0] - 1, unique_labels.shape[0], dtype=int)
#         discrete_colors = colors_tmp[indices, :]
#
#         for n, l in enumerate(unique_labels):
#             mask = (labels == l)
#             _ = ax.set_xlim(emit=False, auto=True)
#             _ = ax.set_ylim(emit=False, auto=True)
#             _ = ax.scatter(embeddings[mask, 0], embeddings[mask, 1], color=discrete_colors[n], label=l, s=size)
#         if len(unique_labels) < 15:
#             _ = ax.legend()
#     else:
#         raise Exception("Exactly one among labels and colors need to be specified.")
#
#     if title:
#         _ = ax.set_title(title)
#     if x_label:
#         _ = ax.set_xlabel(x_label)
#     if y_label:
#         _ = ax.set_ylabel(y_label)
#
#
# def OLD_plot_embeddings(
#         input_dictionary: dict,
#         embedding_key: str,
#         annotation_keys: List[str],
#         x_label: str = None,
#         y_label: str = None,
#         sup_title: str = None,
#         titles: List[str] = None,
#         size: int = 10,
#         n_col: int = 3,
#         cmap_regression: "matplotlib.colors.ListedColormap" = plt.cm.inferno,
#         cmap_labels: "matplotlib.colors.ListedColormap" = plt.cm.tab20,
#         figsize: Tuple[float, float] = None) -> "matplotlib.pyplot.figure":
#
#     def _preprocess_to_numpy(_y) -> numpy.ndarray:
#         if isinstance(_y, torch.Tensor):
#             return _y.cpu().detach().numpy()
#         elif isinstance(_y, list):
#             return numpy.array(_y)
#         elif isinstance(_y, numpy.ndarray):
#             return _y
#         else:
#             raise Exception(
#                 "Labels is either None or torch.Tensor, List, numpy.array. Received {0}".format(type(_y)))
#
#     import scanpy as sc
#     sc.pl.umap
#
#     def _is_color(_y):
#         many_labels = len(numpy.unique(_y)) > 100
#         is_float = isinstance(_y[0].item(), float)
#         return many_labels * is_float
#
#     assert isinstance(annotation_keys, list) and set(annotation_keys).issubset(set(input_dictionary.keys())), \
#         "Error. Annotation_keys must be a list of keys all of which are present in the input dictionary."
#     assert isinstance(embedding_key, str) and embedding_key in input_dictionary.keys(), \
#         "Error. Embedding_key is not present in the input dictionary."
#     assert isinstance(n_col, int) and n_col >= 1, "n_col must be an integer >= 1. Received {0}".format(n_col)
#     n_max = len(annotation_keys)
#
#     assert titles is None or (isinstance(titles, list) and len(titles) == n_max), \
#         "Tiles is either None or a list of length len(annotation_keys) = {0}".format(n_max)
#     assert sup_title is None or isinstance(sup_title, str), \
#         "Sup_tile is either None or a string. Received {0}".format(embedding_key)
#
#     n_col = min(n_col, n_max)
#     n_row = int(numpy.ceil(float(n_max) / n_col))
#     figsize = (4 * n_col, 4 * n_row) if figsize is None else figsize
#
#     embeddings = input_dictionary[embedding_key]
#     assert isinstance(embeddings, torch.Tensor) or isinstance(embeddings, numpy.ndarray), \
#         "Error. Embeddings must be a torch.Tensor or a numpy.ndarray. Received {0}.".format(type(embeddings))
#     embeddings_np = _preprocess_to_numpy(embeddings[:, :2])
#
#     fig, axes = plt.subplots(ncols=n_col, nrows=n_row, figsize=figsize)
#     for n, annotation_k in enumerate(annotation_keys):
#
#         title = None if titles is None else titles[n]
#
#         if n_col == 1 and n_row == 1:
#             ax_curr = axes
#         elif n_row == 1:
#             ax_curr = axes[n]
#         else:
#             c = n % n_col
#             r = n // n_col
#             ax_curr = axes[r, c]
#
#         annotation_tmp = input_dictionary[annotation_k]
#         annotation_np = _preprocess_to_numpy(annotation_tmp)
#
#         if not _is_color(annotation_np):
#             _ = _plot_single_embeddings(
#                 ax=ax_curr,
#                 embeddings=embeddings_np,
#                 x_label=x_label,
#                 y_label=y_label,
#                 title=title,
#                 size=size,
#                 colors=None,
#                 labels=annotation_np,
#                 cmap_colors=None,
#                 cmap_labels=cmap_labels)
#         else:
#             _ = _plot_single_embeddings(
#                 ax=ax_curr,
#                 embeddings=embeddings_np,
#                 x_label=x_label,
#                 y_label=y_label,
#                 title=title,
#                 size=size,
#                 colors=annotation_np,
#                 labels=None,
#                 cmap_colors=cmap_regression,
#                 cmap_labels=None)
#
#     if sup_title:
#         _ = fig.suptitle(sup_title)
#     plt.close(fig)
#     return fig
#