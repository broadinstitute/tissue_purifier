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

    #: long tensor with the total_counts of shape (n).
    #: It is not necessarily equal to counts.sum(dim=-1) since genes can be filtered
    total_counts: torch.Tensor

    #: number of cell types
    k_cell_types: int


def make_gene_dataset_from_anndata(anndata: AnnData, cell_type_key: str, covariate_key: str) -> GeneDataset:
    """
    Convert a anndata object into a GeneDataset object which can be used for gene regression

    Args:
        anndata: AnnData object with the count data
        cell_type_key: key corresponding to the cell type, i.e. cell_type = anndata.obs[cell_type_key]
        covariate_key: key corresponding to the covariate, i.e. covariate = anndata.obsm[covariate_key]

    Returns:
        a GeneDataset object
    """

    cell_types = list(anndata.obs[cell_type_key].values)
    cell_type_ids_n = _make_labels(cell_types)
    counts_ng = torch.tensor(anndata.X.toarray()).long()
    total_counts_n = torch.tensor(anndata.obs['total_counts']).long()
    covariates_nl = torch.tensor(anndata.obsm[covariate_key])

    assert len(counts_ng.shape) == 2
    assert len(covariates_nl.shape) == 2
    assert len(cell_type_ids_n.shape) == 1

    assert counts_ng.shape[0] == covariates_nl.shape[0] == cell_type_ids_n.shape[0]

    k_cell_types = cell_type_ids_n.max().item() + 1  # +1 b/c ids start from zero

    return GeneDataset(
        cell_type_ids=cell_type_ids_n.detach().cpu(),
        covariates=covariates_nl.detach().cpu(),
        total_counts=total_counts_n.detach().cpu(),
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
        total_counts=counts_ng.sum(dim=-1),
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
        arrays = [data.covariates, data.cell_type_ids, data.counts, data.total_counts]
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


class LogNormalPoisson(TorchDistribution):
    """
    A Poisson distribution with rate = N * (log_mu + noise).exp()
    where noise is normally distributed with mean zero and variance sigma, i.e. noise ~ N(0, sigma)

    See http://people.ee.duke.edu/~lcarin/Mingyuan_ICML_2012.pdf
    for discussion of the nice properties of the LogNormalPoisson model
    """

    arg_constraints = {
        "n_trials": constraints.positive,
        "log_rate": constraints.real,
        "noise_scale": constraints.positive,
    }
    support = constraints.nonnegative_integer

    def __init__(
            self,
            n_trials: torch.Tensor,
            log_rate: torch.Tensor,
            noise_scale: torch.Tensor,
            *,
            num_quad_points=8,
            validate_args=None, ):
        """
        Args:
            n_trials: non-negative number of Poisson trials.
            log_rate: the log_rate of a single trial
            noise_scale: controls the level of the injected noise in the log_rate
            num_quad_points: Number of quadrature points used to compute the (approximate) `log_prob`. Defaults to 8.
        """

        if num_quad_points < 1:
            raise ValueError("num_quad_points must be positive.")

        n_trials, log_rate, noise_scale = broadcast_all(
            n_trials, log_rate, noise_scale
        )

        self.quad_points, self.log_weights = get_quad_rule(num_quad_points, log_rate)
        quad_log_rate = (
                log_rate.unsqueeze(-1)
                + noise_scale.unsqueeze(-1) * self.quad_points
        )
        quad_rate = quad_log_rate.exp()
        assert torch.all(torch.isfinite(quad_rate)), "Quad_Rate is not finite."
        assert torch.all(n_trials > 0), "n_trials must be positive"
        assert n_trials.device == quad_rate.device, "Got {0} and {1}".format(n_trials.device, quad_rate.device)
        self.poi_dist = dist.Poisson(rate=n_trials.unsqueeze(-1) * quad_rate)

        self.n_trials = n_trials
        self.log_rate = log_rate
        self.noise_scale = noise_scale
        self.num_quad_points = num_quad_points

        batch_shape = broadcast_shape(
            noise_scale.shape, self.poi_dist.batch_shape[:-1]
        )
        event_shape = torch.Size()
        super().__init__(batch_shape, event_shape, validate_args)

    def log_prob(self, value):
        poi_log_prob = self.poi_dist.log_prob(value.unsqueeze(-1))
        return torch.logsumexp(self.log_weights + poi_log_prob, axis=-1)

    def sample(self, sample_shape=torch.Size()):
        eps = dist.Normal(loc=0, scale=self.noise_scale).sample(sample_shape=sample_shape)
        return dist.Poisson(rate=self.n_trials * torch.exp(self.log_rate + eps)).sample()

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(type(self), _instance)
        batch_shape = torch.Size(batch_shape)

        n_trials = self.n_trials.expand(batch_shape)
        log_rate = self.log_rate.expand(batch_shape)
        noise_scale = self.noise_scale.expand(batch_shape)
        LogNormalPoisson.__init__(
            new,
            n_trials,
            log_rate,
            noise_scale,
            num_quad_points=self.num_quad_points,
            validate_args=False,
        )
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def mean(self):
        return self.n_trials * torch.exp(
            self.log_rate
            + 0.5 * self.noise_scale.pow(2.0)
        )

    @lazy_property
    def variance(self):
        kappa = torch.exp(self.noise_scale.pow(2.0))
        return self.mean + self.mean.pow(2.0) * (kappa - 1.0)


class GeneRegression:
    """
    Given the cell type and some covariates the model predicts the gene expression.

    For each gene, the counts are modelled as a Poisson process with rate: N * (log_mu + noise).exp()
    where:

    N is the total_umi in the cell,

    noise ~ N(0, sigma) with sigma being a gene-specific overdispersion

    log_mu = alpha0 + alpha * covariates

    Notes:
         alpha and alpha0 depend on the cell_type only. Covariates are cell specific an include information
         like the cellular micro-environment
    """

    def __init__(self):
        self._optimizer = None
        self._optimizer_initial_state = None
        self._loss_history = []

    def _model(self,
               dataset: GeneDataset,
               eps_g_range: Tuple[float, float],
               alpha_scale: float,
               subsample_size_cells: int,
               subsample_size_genes: int,
               **kargs):

        # Unpack the dataset
        assert eps_g_range[1] > eps_g_range[0] > 0
        counts_ng = dataset.counts.long()
        total_umi_n = dataset.total_counts.long()
        covariates_nl = dataset.covariates.float()
        cell_type_ids_n = dataset.cell_type_ids.long()  # ids: 0,1,...,K-1
        k = dataset.k_cell_types
        n, g = counts_ng.shape[:2]
        n, l = covariates_nl.shape[:2]
        n = cell_type_ids_n.shape[0]

        # Define the right device:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Define the plates (i.e. conditional independence). It make sense to subsample only gene and cells.
        cell_plate = pyro.plate("cells", size=n, dim=-3, device=device, subsample_size=subsample_size_cells)
        cell_types_plate = pyro.plate("cell_types", size=k, dim=-3, device=device)
        covariate_plate = pyro.plate("covariate", size=l, dim=-2, device=device)
        gene_plate = pyro.plate("genes", size=g, dim=-1, device=device, subsample_size=subsample_size_genes)

        eps_g = pyro.param("eps_g",
                           0.5 * (eps_g_range[0] + eps_g_range[1]) * torch.ones(g, device=device),
                           constraint=constraints.interval(lower_bound=eps_g_range[0],
                                                           upper_bound=eps_g_range[1]))
        alpha0_k1g = pyro.param("alpha0", torch.zeros((k, 1, g), device=device))

        with gene_plate:
            with cell_types_plate:
                with covariate_plate:
                    if alpha_scale is not None:
                        alpha_klg = pyro.sample("alpha", dist.Normal(loc=0.0, scale=alpha_scale)).to(device)
                    else:
                        # note that alpha change the log_rate. Therefore it must be small
                        alpha_klg = pyro.sample("alpha", dist.Uniform(low=-2.0, high=2.0)).to(device)

        with cell_plate as ind_n:
            cell_ids_sub_n = cell_type_ids_n[ind_n].to(device)
            alpha0_n1g = alpha0_k1g[cell_ids_sub_n]
            alpha_nlg = alpha_klg[cell_ids_sub_n]
            covariate_sub_nl1 = covariates_nl[cell_ids_sub_n].unsqueeze(dim=-1).to(device)
            total_umi_n11 = total_umi_n[ind_n, None, None].to(device)

            # assert total_umi_n11.shape == torch.Size([len(ind_n), 1, 1]), "Got {0}".format(total_umi_n11.shape)
            # assert torch.all(total_umi_n11 > 0)
            # assert covariate_sub_nl1.shape == torch.Size([len(ind_n), covariates, 1]), \
            #    "Got {0}".format(covariate_sub_nl1.shape)

            with gene_plate as ind_g:
                # assert eps_g.shape == torch.Size([g]), "Got {0}".format(eps_g.shape)
                # assert alpha0_n1g.shape == torch.Size([len(ind_n), 1, g]), "Got {0}".format(alpha0_n1g.shape)
                # assert alpha_nlg.shape == torch.Size([len(ind_n), covariates, len(ind_g)]), "Got {0}".format(
                #    alpha_nlg.shape)

                assert alpha0_k1g.device == covariate_sub_nl1.device == alpha_nlg.device, \
                    "Got {0} {1} {2}".format(alpha0_k1g.device, covariate_sub_nl1.device, alpha_nlg.device)
                assert ind_n.device == ind_g.device == counts_ng.device, \
                    "Got {0} {1} {2}".format(ind_n.device, ind_g.device, counts_ng.device)

                log_mu_n1g = alpha0_n1g[..., ind_g] + torch.sum(covariate_sub_nl1 * alpha_nlg, dim=-2, keepdim=True)
                eps_sub_g = eps_g[ind_g]

                pyro.sample("counts",
                            LogNormalPoisson(n_trials=total_umi_n11.to(device),
                                             log_rate=log_mu_n1g.to(device),
                                             noise_scale=eps_sub_g.to(device),
                                             num_quad_points=8),
                            obs=counts_ng[ind_n, None].index_select(dim=-1, index=ind_g).to(device))

    def _guide(self,
               dataset: GeneDataset,
               use_covariates: bool,
               subsample_size_genes: int,
               **kargs):

        # Unpack the dataset
        n, g = dataset.counts.shape[:2]
        n, l = dataset.covariates.shape[:2]
        k = dataset.k_cell_types

        # Define the right device:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Define the gene and cell plates. It make sense to subsample only gene and cells.
        # cell_plate = pyro.plate("cells", size=n, dim=-3, device=device, subsample_size=subsample_size_cells)
        cell_types_plate = pyro.plate("cell_types", size=k, dim=-3, device=device)
        covariate_plate = pyro.plate("covariate", size=l, dim=-2, device=device)
        gene_plate = pyro.plate("genes", size=g, dim=-1, device=device, subsample_size=subsample_size_genes)

        alpha_param_loc_klg = pyro.param("alpha_loc", torch.zeros((k, l, g), device=device))

        with gene_plate as ind_g:
            with cell_types_plate:
                with covariate_plate:
                    if use_covariates:
                        alpha_loc_tmp = alpha_param_loc_klg[..., ind_g]
                    else:
                        alpha_loc_tmp = torch.zeros_like(alpha_param_loc_klg[..., ind_g])
                    alpha_klg = pyro.sample("alpha", dist.Delta(v=alpha_loc_tmp))
                    assert alpha_klg.shape == torch.Size([k, l, len(ind_g)])

    def get_params(self):
        """ Returns a (detached) dictionary with fitted parameters """
        mydict = dict()
        for k, v in pyro.get_param_store().items():
            mydict[k] = v.detach().cpu()
        return mydict

    def predict(self, dataset: GeneDataset, num_samples: int = 10) -> (dict, pd.DataFrame, pd.DataFrame):
        """
        Args:
            dataset: the dataset to run the prediction on
            num_samples: how many random samples to draw from the predictive distribution

        Returns:
            result: a distionary with the true and predicted counts, the cell_type and two metrics (log_score and deviance)
            log_score_df: a DataFrame with the log_score evaluation metric
            deviance_df: a DataFrame with the deviance evaluation metric
        """

        n, g = dataset.counts.shape[:2]
        n, l = dataset.covariates.shape[:2]
        k = dataset.k_cell_types

        eps_g = pyro.get_param_store().get_param("eps_g")
        alpha0_k1g = pyro.get_param_store().get_param("alpha0")
        alpha_klg = pyro.get_param_store().get_param("alpha_loc")
        counts_ng = dataset.counts
        total_umi_n = dataset.total_counts
        cell_type_ids = dataset.cell_type_ids.long()
        covariates_nl1 = dataset.covariates.unsqueeze(dim=-1)

        assert eps_g.shape == torch.Size([g]), \
            "Got {0}. Are you predicting on the right dataset?".format(eps_g.shape)
        assert alpha0_k1g.shape == torch.Size([k, 1, g]), \
            "Got {0}. Are you predicting on the right dataset?".format(alpha0_k1g.shape)
        assert alpha_klg.shape == torch.Size([k, l, g]), \
            "Got {0}. Are you predicting on the right dataset?".format(alpha_klg.shape)

        if torch.cuda.is_available():
            cell_type_ids = cell_type_ids.cuda()
            eps_g = eps_g.cuda()
            alpha0_k1g = alpha0_k1g.cuda()
            alpha_klg = alpha_klg.cuda()
            counts_ng = counts_ng.cuda()
            total_umi_n = total_umi_n.cuda()
            covariates_nl1 = covariates_nl1.cuda()

        log_rate_n1g = alpha0_k1g[cell_type_ids] + (covariates_nl1 * alpha_klg[cell_type_ids]).sum(dim=-2, keepdim=True)
        total_umi_n1 = total_umi_n[:, None]

        mydist = LogNormalPoisson(
            n_trials=total_umi_n1,
            log_rate=log_rate_n1g.squeeze(dim=-2),
            noise_scale=eps_g,
            num_quad_points=8)

        counts_pred_bng = mydist.sample(sample_shape=torch.Size([num_samples]))
        log_score_ng = mydist.log_prob(counts_ng)

        results = {
            "counts_true": counts_ng.detach().cpu(),
            "counts_pred": counts_pred_bng.detach().cpu(),
            "log_score": -log_score_ng.detach().cpu(),
            "deviance": (counts_ng-counts_pred_bng).abs().detach().cpu(),
            "cell_type": cell_type_ids.detach().cpu()
        }

        # package the results into two dataframe for easy visualization
        cols = ["cell_type"] + ['gene_{}'.format(g) for g in range(results["log_score"].shape[-1])]
        c_log_score = torch.cat((results["cell_type"].unsqueeze(dim=-1),
                                 results["log_score"]), dim=-1).numpy()
        log_score_df = pd.DataFrame(c_log_score, columns=cols)
        c_deviance = torch.cat((results["cell_type"].unsqueeze(dim=-1),
                                results["deviance"].mean(dim=-3)), dim=-1).numpy()
        deviance_df = pd.DataFrame(c_deviance, columns=cols)

        return results, log_score_df, deviance_df

    def render_model(self, dataset: GeneDataset, filename: Optional[str] = None,  render_distributions: bool = False):
        """
        Wrapper around :method:'pyro.render_model'.

        Args:
            dataset: dataset to use for computing the shapes
            filename: File to save rendered model in. If None (defaults) the image is displayed.
            render_distributions: Whether to include RV distribution
        """
        model_kargs = {
            'dataset': dataset,
            'use_covariates': True,
            'alpha_scale': None,
            'eps_g_range': (0.01, 0.02),
            'subsample_size_genes': None,
            'subsample_size_cells': None,
        }

        trace = pyro.poutine.trace(self._model).get_trace(**model_kargs)
        trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
        print(trace.format_shapes())

        return pyro.render_model(self._model,
                                 model_kwargs=model_kargs,
                                 filename=filename,
                                 render_distributions=render_distributions)

    @property
    def optimizer(self) -> pyro.optim.PyroOptim:
        assert self._optimizer is not None, "Optimizer is not specified. Call configure_optimizer first."
        return self._optimizer

    def configure_optimizer(self, optimizer_type: str = 'adam', lr: float = 5E-3):
        if optimizer_type == 'adam':
            self._optimizer = pyro.optim.Adam({"lr": lr})
        else:
            raise NotImplementedError

        self._optimizer_initial_state = self._optimizer.get_state()

    def show_loss(self, figsize: Tuple[float, float] = (4, 4), logx: bool = False, logy: bool = False, ax=None):
        """
        Show the loss history. Usefull for checking if the training has converged.

        Args:
            figsize: the size of the image. Used only if ax=None
            logx: if True the x_axis is shown in logarithmic scale
            logy: if True the x_axis is shown in logarithmic scale
            ax: The axes object to draw the plot onto. If None (defaults) creates a new figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ax.plot(self._loss_history)

        if logx:
            ax.set_xscale("log")
        else:
            ax.set_xscale("linear")

        if logy:
            ax.set_yscale("log")
        else:
            ax.set_xscale("linear")

        if ax is None:
            fig.tight_layout()
            plt.close(fig)
            return fig

    def train(self,
              dataset: GeneDataset,
              n_steps: int = 2500,
              print_frequency: int = 50,
              use_covariates: bool = True,
              alpha_scale: float = None,
              eps_g_range: Tuple[float, float] = (1.0E-3, 1.0),
              subsample_size_cells: int = None,
              subsample_size_genes: int = None,
              from_scratch: bool = True):
        """
        Train the model.

        Args:
            dataset: Dataset to train the model on
            n_steps: number of training step
            print_frequency: how frequently to print loss to screen
            use_covariates: if true, use covariates, if false use cell type information only
            alpha_scale: controlls the strength of the L2 regularization on the regression coefficients.
                The strength is proportional to 1/alpha_scale. Therefore small alpha means strong regularization.
                If None (defaults) there is no regularization.
            eps_g_range: range on possible values of the gene-specific noise. Must the a strictly positive range.
            subsample_size_genes: for large dataset, the minibatch can be created using a subset of genes.
            subsample_size_cells: for large dataset, the minibatch can be created using a subset of cells.
            from_scratch: it True (defaults) the training start from scratch. If False the training continues
                from where it was left off. Usefull for extending a previously started training.

        Note:
            If you get an out-of-memory error try to tune the subsample_size_cells and subsample_size_genes
        """

        if from_scratch:
            pyro.clear_param_store()
            self._loss_history = []
            assert self.optimizer is not None, "Optimizer is not specified. Call configure_optimizer first."
            self.optimizer.set_state(self._optimizer_initial_state)

        model_kargs = {
            'dataset': dataset,
            'use_covariates': use_covariates,
            'alpha_scale': alpha_scale,
            'eps_g_range': eps_g_range,
            'subsample_size_genes': subsample_size_genes,
            'subsample_size_cells': subsample_size_cells,
        }

        svi = SVI(self._model, self._guide, self.optimizer, loss=Trace_ELBO())
        for i in range(n_steps+1):
            loss = svi.step(**model_kargs)
            self._loss_history.append(loss)
            if i % print_frequency == 0:
                print('[iter {}]  loss: {:.4f}'.format(i, loss))

    def save_ckpt(self, filename: str):
        """ Save the full state of the model and optimizer to disk """
        ckpt = {
            "param_store": pyro.get_param_store().get_state(),
            "optimizer": self._optimizer,
            "optimizer_state": self._optimizer.get_state(),
            "optimizer_initial_state": self._optimizer_initial_state,
            "loss_history": self._loss_history
        }

        with open(filename, "wb") as output_file:
            torch.save(ckpt, output_file)

    def load_ckpt(self, filename: str, map_location=None):
        """ Load the full state of the model and optimizer from disk """

        with open(filename, "rb") as input_file:
            ckpt = torch.load(input_file, map_location)

        pyro.clear_param_store()
        pyro.get_param_store().set_state(ckpt["param_store"])
        self._optimizer = ckpt["optimizer"]
        self._optimizer.set_state(ckpt["optimizer_state"])
        self._optimizer_initial_state = ckpt["optimizer_initial_state"]
        self._loss_history = ckpt["loss_history"]


### def EMD_between_distributions(distA, distB, normalize: bool = False):
###     """
###     Eearth mover's distance (aka  Wasserstein distance) has a close form solution in 1D.
###     See https://en.wikipedia.org/wiki/Wasserstein_metric)
###     """
###
###     sizeA = distA.shape[-1]
###     sizeB = distB.shape[-1]
###     max_size = max(sizeA, sizeB)
###     min_size = min(sizeA, sizeB)
###     delta_size = max_size - min_size
###
###     padder = torch.nn.ConstantPad1d(padding=(0, delta_size), value=0)
###     _distA = padder(distA)[..., :max_size]
###     _distB = padder(distB)[..., :max_size]
###
###     if normalize:
###         normA = _distA.sum(dim=-1, keepdim=True)
###         normB = _distB.sum(dim=-1, keepdim=True)
###         _distA /= normA
###         _distB /= normB
###
###     # Actual caltulation
###     _distA_cum = torch.cumsum(_distA, axis=-1)
###     _distB_cum = torch.cumsum(_distB, axis=-1)
###     EMD = (_distA_cum - _distB_cum).abs().sum(axis=-1)
###     return EMD
###
###
### def L1_between_distributions(distA, distB, normalize: bool = False):
###     """ Simple L1 distance between two distributions. """
###     sizeA = distA.shape[-1]
###     sizeB = distB.shape[-1]
###     max_size = max(sizeA, sizeB)
###     min_size = min(sizeA, sizeB)
###     delta_size = max_size - min_size
###
###     padder = torch.nn.ConstantPad1d(padding=(0, delta_size), value=0)
###     _distA = padder(distA)[..., :max_size]
###     _distB = padder(distB)[..., :max_size]
###
###     if normalize:
###         normA = _distA.sum(dim=-1, keepdim=True)
###         normB = _distB.sum(dim=-1, keepdim=True)
###         _distA /= normA
###         _distB /= normB
###
###     # Actual calculation
###     L1_norm = (_distA - _distB).abs().sum(axis=-1)
###     return L1_norm
###

# This guide does MAP. Everything is a parameter except eps_n1g
### def guide_MAP(dataset, observed: bool = True, use_covariates: bool = True):
###     # Everything is point estimate
###     # Unpack the dataset
###     counts_ng = dataset["counts"].long()
###     covariates_nl = dataset['other_covariates'].float()
###     cell_type_ids_n = dataset['cell_type_codes'].long()  # ids: 0,1,...,K-1
###     k = cell_type_ids_n.max().item() + 1
###
###     n, g = counts_ng.shape[:2]
###     n, l = covariates_nl.shape[:2]
###     n = cell_type_ids_n.shape[0]
###     assert isinstance(k, int) and k > 0, "Got {0}".format(k)
###     assert isinstance(l, int) and l > 0, "Got {0}".format(l)
###     assert isinstance(n, int) and n > 0, "Got {0}".format(n)
###     assert isinstance(g, int) and g > 0, "Got {0}".format(g)
###
###     if torch.cuda.is_available():
###         covariates_nl = covariates_nl.cuda()
###         cell_type_ids_n = cell_type_ids_n.cuda()
###         counts_ng = counts_ng.cuda()
###     device = covariates_nl.device
###
###     # Define the gene and cell plates. It make sense to subsample only gene and cells.
###     cell_plate = pyro.plate("cells", size=n, dim=-3, device=device, subsample_size=None)
###     cell_types_plate = pyro.plate("cell_types", size=k, dim=-3, device=device)
###     covariate_plate = pyro.plate("covariate", size=l, dim=-2, device=device)
###     gene_plate = pyro.plate("genes", size=g, dim=-1, device=device, subsample_size=None)
###
###     eps_param_g = pyro.param(
###         "eps_loc_g",
###         eps_g_low * torch.ones(g, device=device),
###         constraint=constraints.interval(eps_g_low, eps_g_high))
###
###     alpha_param_loc_klg = pyro.param(
###         "alpha_loc",
###         torch.zeros((k, l, g), device=device))
###
###     eps_param_loc_n1g = pyro.param(
###         "eps_loc_n1g",
###         torch.zeros((n, 1, g), device=device))
###
###     with gene_plate as ind_g:
###         eps_g = pyro.sample("eps_g", dist.Delta(v=eps_param_g[ind_g]))
###         with cell_types_plate:
###             with covariate_plate:
###                 v = alpha_param_loc_klg[..., ind_g]
###                 if use_covariates:
###                     alpha_klg = pyro.sample("alpha", dist.Delta(v=v))
###                 else:
###                     alpha_klg = pyro.sample("alpha", dist.Delta(v=torch.zeros_like(v)))
###
###     with cell_plate as ind_n:
###         with gene_plate as ind_g:
###             eps_n1g = pyro.sample("eps_n1g", dist.Delta(v=eps_param_loc_n1g[ind_n][..., ind_g]))
###
### # Everything is point estimate except eps_n1g
### def guide_eps(dataset, observed: bool = True, use_covariates: bool = True):
###     # Unpack the dataset
###     counts_ng = dataset["counts"].long()
###     covariates_nl = dataset['other_covariates'].float()
###     cell_type_ids_n = dataset['cell_type_codes'].long()  # ids: 0,1,...,K-1
###     k = cell_type_ids_n.max().item() + 1
###
###     n, g = counts_ng.shape[:2]
###     n, l = covariates_nl.shape[:2]
###     n = cell_type_ids_n.shape[0]
###     assert isinstance(k, int) and k > 0, "Got {0}".format(k)
###     assert isinstance(l, int) and l > 0, "Got {0}".format(l)
###     assert isinstance(n, int) and n > 0, "Got {0}".format(n)
###     assert isinstance(g, int) and g > 0, "Got {0}".format(g)
###
###     if torch.cuda.is_available():
###         covariates_nl = covariates_nl.cuda()
###         cell_type_ids_n = cell_type_ids_n.cuda()
###         counts_ng = counts_ng.cuda()
###     device = covariates_nl.device
###
###     # Define the gene and cell plates. It make sense to subsample only gene and cells.
###     cell_plate = pyro.plate("cells", size=n, dim=-3, device=device, subsample_size=None)
###     cell_types_plate = pyro.plate("cell_types", size=k, dim=-3, device=device)
###     covariate_plate = pyro.plate("covariate", size=l, dim=-2, device=device)
###     gene_plate = pyro.plate("genes", size=g, dim=-1, device=device, subsample_size=None)
###
###     eps_param_g = pyro.param(
###         "eps_loc_g",
###         eps_g_low * torch.ones(g, device=device),
###         constraint=constraints.interval(eps_g_low, eps_g_high))
###
###     alpha_param_loc_klg = pyro.param(
###         "alpha_loc",
###         torch.zeros((k, l, g), device=device))
###
###     with gene_plate as ind_g:
###         eps_g = pyro.sample("eps_g", dist.Delta(v=eps_param_g[ind_g]))
###         with cell_types_plate:
###             with covariate_plate:
###                 v = alpha_param_loc_klg[..., ind_g]
###                 if use_covariates:
###                     alpha_klg = pyro.sample("alpha", dist.Delta(v=v))
###                 else:
###                     alpha_klg = pyro.sample("alpha", dist.Delta(v=torch.zeros_like(v)))
###
###     with cell_plate as ind_n:
###         with gene_plate as ind_g:
###             eps_n1g = pyro.sample("eps_n1g", dist.Normal(loc=0, scale=eps_g[ind_g]))
###
###
### # Everything is point estimate except eps_n1g, alpha
### def guide_eps_alpha(dataset, observed: bool = True, use_covariates: bool = True):
###     # Unpack the dataset
###     counts_ng = dataset["counts"].long()
###     covariates_nl = dataset['other_covariates'].float()
###     cell_type_ids_n = dataset['cell_type_codes'].long()  # ids: 0,1,...,K-1
###     k = cell_type_ids_n.max().item() + 1
###
###     n, g = counts_ng.shape[:2]
###     n, l = covariates_nl.shape[:2]
###     n = cell_type_ids_n.shape[0]
###     assert isinstance(k, int) and k > 0, "Got {0}".format(k)
###     assert isinstance(l, int) and l > 0, "Got {0}".format(l)
###     assert isinstance(n, int) and n > 0, "Got {0}".format(n)
###     assert isinstance(g, int) and g > 0, "Got {0}".format(g)
###
###     if torch.cuda.is_available():
###         covariates_nl = covariates_nl.cuda()
###         cell_type_ids_n = cell_type_ids_n.cuda()
###         counts_ng = counts_ng.cuda()
###     device = covariates_nl.device
###
###     # Define the gene and cell plates. It make sense to subsample only gene and cells.
###     cell_plate = pyro.plate("cells", size=n, dim=-3, device=device, subsample_size=None)
###     cell_types_plate = pyro.plate("cell_types", size=k, dim=-3, device=device)
###     covariate_plate = pyro.plate("covariate", size=l, dim=-2, device=device)
###     gene_plate = pyro.plate("genes", size=g, dim=-1, device=device, subsample_size=None)
###
###     eps_param_g = pyro.param(
###         "eps_loc_g",
###         eps_g_low * torch.ones(g, device=device),
###         constraint=constraints.interval(eps_g_low, eps_g_high))
###
###     alpha_param_loc_klg = pyro.param(
###         "alpha_loc",
###         torch.zeros((k, l, g), device=device))
###
###     alpha_param_scale_klg = pyro.param(
###         "alpha_scale",
###         alpha_scale * torch.ones((k, l, g), device=device),
###         constraint=constraints.positive)
###
###     with gene_plate as ind_g:
###         eps_g = pyro.sample("eps_g", dist.Delta(v=eps_param_g[ind_g]))
###         with cell_types_plate:
###             with covariate_plate:
###                 loc = alpha_param_loc_klg[..., ind_g]
###                 scale = alpha_param_scale_klg[..., ind_g]
###                 if use_covariates:
###                     alpha_klg = pyro.sample("alpha", dist.Normal(loc=loc, scale=scale))
###                 else:
###                     alpha_klg = pyro.sample("alpha", dist.Delta(v=torch.zeros_like(loc)))
###
###     with cell_plate as ind_n:
###         with gene_plate as ind_g:
###             eps_n1g = pyro.sample("eps_n1g", dist.Normal(loc=0, scale=eps_g[ind_g]))
###