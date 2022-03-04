from typing import NamedTuple, Tuple, Dict, Optional, Union, List, Any
import torch
import pyro
import pyro.distributions as dist
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import broadcast_shape
from pyro.ops.special import get_quad_rule
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all, lazy_property
from pyro.infer import SVI, Trace_ELBO
import pandas as pd
import pyro.poutine
import pyro.optim
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import matplotlib.pyplot as plt
import numpy
from scanpy import AnnData


def _make_labels(y: Union[torch.Tensor, numpy.ndarray, List[Any]]) -> torch.Tensor:
    def _to_numpy(_y):
        if isinstance(_y, numpy.ndarray):
            return _y
        elif isinstance(_y, list):
            return numpy.array(_y)
        elif isinstance(_y, torch.Tensor):
            return _y.detach().cpu().numpy()

    y_np = _to_numpy(y)
    unique_labels = numpy.unique(y_np)
    y_to_ids = dict(zip(unique_labels, numpy.arange(y_np.shape[0])))
    labels = torch.tensor([y_to_ids[tmp] for tmp in y_np])
    return labels


class GeneDataset(NamedTuple):
    """
    Container for organizing the gene expression data
    """
    # order is important. Do not change

    #: float tensor with the covariates of shape (n, k)
    covariates: torch.Tensor

    #: long tensor with the cell_type_ids of shape (n)
    cell_type_ids: torch.Tensor

    #: long tensor with the count data of shape (n, g)
    counts: torch.Tensor

    #: number of cell types
    k_cell_types: int


def make_gene_dataset_from_anndata(
        anndata: AnnData,
        cell_type_key: str,
        covariate_key: str) -> GeneDataset:
    """
    Convert a anndata object into a GeneDataset object which can be used for gene regression

    Args:
        anndata: AnnData object with the count data
        cell_type_key: key corresponding to the cell type, i.e. cell_types = anndata.obs[cell_type_key]
        covariate_key: key corresponding to the covariate, i.e. covariates = anndata.obsm[covariate_key]

    Returns:
        a GeneDataset object
    """

    cell_types = list(anndata.obs[cell_type_key].values)
    cell_type_ids_n = _make_labels(cell_types)
    counts_ng = torch.tensor(anndata.X.toarray()).long()
    covariates_nl = torch.tensor(anndata.obsm[covariate_key])

    assert len(counts_ng.shape) == 2
    assert len(covariates_nl.shape) == 2
    assert len(cell_type_ids_n.shape) == 1

    assert counts_ng.shape[0] == covariates_nl.shape[0] == cell_type_ids_n.shape[0]

    k_cell_types = cell_type_ids_n.max().item() + 1  # +1 b/c ids start from zero

    return GeneDataset(
        cell_type_ids=cell_type_ids_n.detach().cpu(),
        covariates=covariates_nl.detach().cpu(),
        counts=counts_ng.detach().cpu(),
        k_cell_types=k_cell_types)


def generate_fake_data(
        cells: int = 20000,
        genes: int = 500,
        covariates: int = 20,
        cell_types: int = 9,
        alpha_scale: float = 0.01,
        alpha0_loc: float = -5.0,
        alpha0_scale: float = 0.5,
        noise_scale: float = 0.1):
    """
    Helper function to generate synthetic count data

    Args:
        cells: number of cells
        genes: number of genes
        covariates: number of covariates
        cell_types: number of cell types
        alpha_scale: scale for alpha (i.e. the regression coefficients)
        alpha0_loc: loc of alpha0 (mean value for the zero regression coefficients, i.e. offset)
        alpha0_scale: scale for alpha0 (i.e. variance of the zero regression coefficients, i.e. offset)
        noise_scale: noise scale (gene-specific overdispersion)
    """
    cov_nl = torch.randn((cells, covariates))
    cell_ids_n = torch.randint(low=0, high=cell_types, size=[cells])

    alpha_klg = alpha_scale * torch.randn((cell_types, covariates, genes))
    alpha0_kg = alpha0_loc + alpha0_scale * torch.randn((cell_types, genes))
    eps_g = torch.randn(genes).abs() * noise_scale + 1E-4  # std per gene
    eps_ng = torch.randn(cells, genes) * eps_g

    log_mu_ng = alpha0_kg[cell_ids_n] + (cov_nl[..., None] * alpha_klg[cell_ids_n]).sum(dim=-2)
    mu_ng = (log_mu_ng + eps_ng).exp()
    rate_ng = torch.randint(low=250, high=3000, size=[cells, 1]) * mu_ng
    counts_ng = torch.poisson(rate_ng).long()

    # simple Q/A on the fake data
    total_umi_n = counts_ng.sum(dim=-1)
    eps_g_low = torch.min(eps_g.abs()).item()
    eps_g_high = torch.max(eps_g.abs()).item()
    assert eps_g_high > eps_g_low > 0, "Error. Got {0}, {1}".format(eps_g_high, eps_g_low)
    assert len(counts_ng.shape) == 2, "Error. Got {0}".format(counts_ng.shape)
    assert len(cov_nl.shape) == 2, "Error. Got {0}".format(cov_nl.shape)
    assert len(cell_ids_n.shape) == 1, "Error. Got {0}".format(cell_ids_n.shape)
    assert counts_ng.shape[0] == cov_nl.shape[0] == cell_ids_n.shape[0],  \
        "Error. Got {0} {1} {2}".format(counts_ng.shape, cov_nl.shape, cell_ids_n.shape)
    assert torch.all(total_umi_n > 0),  \
        "Error. Some elements are zero, negative or inf {0}".format(torch.all(total_umi_n >= 0))

    return GeneDataset(
        counts=counts_ng,
        covariates=cov_nl,
        cell_type_ids=cell_ids_n,
        k_cell_types=cell_types)


def train_test_val_split(
        data: Union[List[torch.Tensor], List[numpy.ndarray], GeneDataset],
        train_size: float = 0.8,
        test_size: float = 0.15,
        val_size: float = 0.05,
        n_splits: int = 1,
        random_state: int = None,
        stratify: bool = True):
    """
    Args:
        data: the data to split into train/test/val
        train_size: the relative size of the train dataset
        test_size: the relative size of the test dataset
        val_size: the relative size of the val dataset
        n_splits: how many times to split the data
        random_state: specify the random state for reproducibility
        stratify: If true the tran/test/val are stratified so that they contain approximately the same
            number of example from each class. If data is a list of arrays the 2nd array is assumed to represent the
            class. If data is a GeneDataset the class is the cell_type.

    Returns:
        yields multiple splits of the data.

    Example:
          >>> for train, test, val in train_test_val_split(data=[x,y,z]):
          >>>       x_train, y_train, z_train = train
          >>>       x_test, y_test, z_test = test
          >>>       x_val, y_val, z_val = val
          >>>       ... do something ...

    Example:
          >>> for train, test, val in train_test_val_split(data=GeneDataset):
          >>>       assert isinstance(train, GeneDataset)
          >>>       assert isinstance(test, GeneDataset)
          >>>       assert isinstance(val, GeneDataset)
          >>>       ... do something ...
    """

    if train_size <= 0:
        raise ValueError("Train_size must be > 0")
    if test_size <= 0:
        raise ValueError("Test_size must be > 0")
    if val_size <= 0:
        raise ValueError("Val_size must be > 0")

    if isinstance(data, List):
        arrays = data
    elif isinstance(data, GeneDataset):
        # same order as in the definition of GeneDataset NamedTuple
        arrays = [data.covariates, data.cell_type_ids, data.counts]
    else:
        raise ValueError("data must be a list or a GeneDataset")

    dims_actual = [a.shape[0] for a in arrays]
    dims_expected = [dims_actual[0]] * len(dims_actual)
    assert all(a == b for a, b in zip(dims_actual, dims_expected)), \
        "Error. All leading dimensions should be the same"

    # Normalize the train/test/val sizes
    norm0 = train_size + test_size + val_size
    train_size_norm0 = train_size / norm0
    test_and_val_size_norm0 = (test_size + val_size) / norm0

    norm1 = test_size + val_size
    test_size_norm1 = test_size / norm1
    val_size_norm1 = val_size / norm1

    if stratify:
        sss0 = StratifiedShuffleSplit(n_splits=n_splits,
                                      train_size=train_size_norm0,
                                      test_size=test_and_val_size_norm0,
                                      random_state=random_state)
        sss1 = StratifiedShuffleSplit(n_splits=1,
                                      train_size=val_size_norm1,
                                      test_size=test_size_norm1,
                                      random_state=random_state)
    else:
        sss0 = ShuffleSplit(n_splits=n_splits,
                            train_size=train_size_norm0,
                            test_size=test_and_val_size_norm0,
                            random_state=random_state)
        sss1 = ShuffleSplit(n_splits=1,
                            train_size=val_size_norm1,
                            test_size=test_size_norm1,
                            random_state=random_state)

    # Part common to both stratified and not stratified
    trains, tests, vals = None, None, None
    for index_train, index_test_and_val in sss0.split(*arrays[:2]):
        trains = [a[index_train] for a in arrays]
        test_and_val = [a[index_test_and_val] for a in arrays]
        for index_val, index_test in sss1.split(*test_and_val[:2]):
            tests = [a[index_test] for a in test_and_val]
            vals = [a[index_val] for a in test_and_val]

        if isinstance(data, GeneDataset):
            k = data.k_cell_types
            trains.append(k)
            tests.append(k)
            vals.append(k)
            yield GeneDataset._make(trains), GeneDataset._make(tests), GeneDataset._make(vals)
        else:
            yield trains, tests, vals
