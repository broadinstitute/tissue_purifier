from typing import NamedTuple, List, Dict, Callable
import torch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints
from pyro.infer import SVI, Trace_ELBO
from tissue_purifier.model_utils.log_poisson_dist import LogNormalPoisson
import pyro.optim
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit


class GeneDataset(NamedTuple):
    counts: torch.Tensor  # shape: n,g
    covariates: torch.Tensor  # shape: n,l
    cell_type_ids: torch.Tensor  # shape: n
    k_cell_types: int
    eps_g_low: float
    eps_g_high: float
    alpha_scale: float


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
    Args:
        cells: number of cells
        genes: number of genes
        covariates: number of covariates
        cell_types: number of cell types
        alpha_scale: scale for alpha
        alpha0_loc: loc of alpha0
        alpha0_scale: scale for alpha0
        noise_scale: noise scale
    """
    cov_nl = torch.randn((cells, covariates))
    cell_ids_n = torch.randint(low=0, high=cell_types, size=[cells])

    alpha_klg = alpha_scale * torch.randn((cell_types, covariates, genes))
    alpha0_kg = alpha0_loc + alpha0_scale * torch.randn((cell_types, genes))
    eps_g = torch.randn(genes) * noise_scale  # std per gene
    eps_ng = torch.randn(cells, genes) * eps_g

    log_mu_ng = alpha0_kg[cell_ids_n] + (cov_nl[..., None] * alpha_klg[cell_ids_n]).sum(dim=-2)
    mu_ng = (log_mu_ng + eps_ng).exp()
    rate_ng = torch.randint(low=250, high=3000, size=[cells, 1]) * mu_ng
    counts_ng = torch.poisson(rate_ng).long()

    # simple Q/A on the fake data
    total_umi_n = counts_ng.sum(dim=-1)
    eps_g_low = torch.min(eps_ng.abs()).item()
    eps_g_high = torch.max(eps_ng.abs()).item()
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
        k_cell_types=cell_types,
        eps_g_low=eps_g_low * 0.5,
        eps_g_high=eps_g_high,
        alpha_scale=alpha_scale)


def train_test_val_split(
    *arrays,
    train_size: float = 0.8,
    test_size: float = 0.15,
    val_size: float = 0.05,
    random_state=None,
    stratify=True,
    ):
    """
    Returns 3 lists with the train,test and validation tensors.
    The length of each list depends on the number of tensors passed in.

    For example:

    [X_train, y_train, z_train], [X_test, y_test, z_test], [X_val, y_val, z_val] =
    train_test_val_split(X, y, z, train_size=0.8, test_size=0.1, val_size=0.0, random_state=42)

    Note:
        The stratification (if applicable) is done on the second tensor (y in the example above)
    """
    dims_actual = [a.shape[0] for a in arrays]
    dims_expected = [dims_actual[0]] * len(dims_actual)
    assert all(a == b for a, b in zip(dims_actual, dims_expected)), \
        "Error. All leading dimensions should be the same"

    norm0 = train_size + test_size + val_size
    train_size_norm0 = train_size / norm0
    test_and_val_size_norm0 = (test_size + val_size) / norm0

    norm1 = test_size + val_size
    test_size_norm1 = test_size / norm1
    val_size_norm1 = val_size / norm1

    if stratify:
        sss0 = StratifiedShuffleSplit(n_splits=1,
                                      train_size=train_size_norm0,
                                      test_size=test_and_val_size_norm0,
                                      random_state=random_state)
        sss1 = StratifiedShuffleSplit(n_splits=1,
                                      train_size=val_size_norm1,
                                      test_size=test_size_norm1,
                                      random_state=random_state)
    else:
        sss0 = ShuffleSplit(n_splits=1,
                            train_size=train_size_norm0,
                            test_size=test_and_val_size_norm0,
                            random_state=random_state)
        sss1 = ShuffleSplit(n_splits=1,
                            train_size=val_size_norm1,
                            test_size=test_size_norm1,
                            random_state=random_state)

    # Part common to both stratified and not stratified
    for index_train, index_test_and_val in sss0.split(*arrays):
        trains = [a[index_train] for a in arrays]
        test_and_val = [a[index_test_and_val] for a in arrays]
        for index_val, index_test in sss1.split(*test_and_val):
            tests = [a[index_test] for a in test_and_val]
            vals = [a[index_val] for a in test_and_val]

    return trains, tests, vals


def train_helper(
        model: Callable = None,
        guide: Callable = None,
        model_args: List = [],
        model_kargs: Dict = dict(),
        optimizer: pyro.optim.PyroOptim = pyro.optim.Adam({"lr": 0.005}),
        n_steps: int = 2500,
        print_frequency: int = 50,
        clear_param_store=True):

    if clear_param_store:
        pyro.clear_param_store()

    if model is None:
        model = model_poisson_log_normal
    if guide is None:
        guide = guide_poisson_log_normal

    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    for i in range(n_steps):
        loss = svi.step(*model_args, **model_kargs)
        if i % print_frequency == 0:
            print('[iter {}]  loss: {:.4f}'.format(i, loss))


def model_poisson_log_normal(
        dataset: GeneDataset,
        observed: bool = True,
        regularize_alpha: bool = True,
        subsample_size_cells: int = None,
        subsample_size_genes: int = None,
        **kargs):

    # Unpack the dataset
    counts_ng = dataset.counts.long()
    covariates_nl = dataset.covariates.float()
    cell_type_ids_n = dataset.cell_type_ids.long()  # ids: 0,1,...,K-1
    k = dataset.k_cell_types
    n, g = counts_ng.shape[:2]
    n, covariates = covariates_nl.shape[:2]
    n = cell_type_ids_n.shape[0]  

    if torch.cuda.is_available(): 
        covariates_nl = covariates_nl.cuda()
        cell_type_ids_n = cell_type_ids_n.cuda()
        counts_ng = counts_ng.cuda()
    device = covariates_nl.device
    
    # Define the plates. It make sense to subsample only gene and cells.
    cell_plate = pyro.plate("cells", size=n, dim=-3, device=device, subsample_size=subsample_size_cells)
    cell_types_plate = pyro.plate("cell_types", size=k, dim=-3, device=device)
    covariate_plate = pyro.plate("covariate", size=covariates, dim=-2, device=device)
    gene_plate = pyro.plate("genes", size=g, dim=-1, device=device, subsample_size=subsample_size_genes)

    eps_g = pyro.param("eps_g",
                       dataset.eps_g_low*torch.ones(g, device=device),
                       constraint=constraints.interval(lower_bound=dataset.eps_g_low, upper_bound=dataset.eps_g_high))
    alpha0_k1g = pyro.param("alpha0", torch.zeros((k, 1, g), device=device))

    with gene_plate:
        with cell_types_plate:
            with covariate_plate:
                if regularize_alpha:
                    alpha_klg = pyro.sample("alpha", dist.Normal(loc=0.0, scale=dataset.alpha_scale)).to(device)
                else:
                    # note that alpha change the log_rate. Therefore it must be small
                    alpha_klg = pyro.sample("alpha", dist.Uniform(low=-2.0, high=2.0)).to(device)

    with cell_plate as ind_n:
        cell_ids_sub_n = cell_type_ids_n[ind_n]
        alpha0_n1g = alpha0_k1g[cell_ids_sub_n]
        alpha_nlg = alpha_klg[cell_ids_sub_n]
        covariate_sub_nl1 = covariates_nl[cell_ids_sub_n].unsqueeze(dim=-1)
        total_umi_n11 = counts_ng[ind_n].sum(dim=-1, keepdim=True).unsqueeze(dim=-1)

        # assert total_umi_n11.shape == torch.Size([len(ind_n), 1, 1]), "Got {0}".format(total_umi_n11.shape)
        # assert torch.all(total_umi_n11 > 0)
        # assert covariate_sub_nl1.shape == torch.Size([len(ind_n), covariates, 1]), \
        #    "Got {0}".format(covariate_sub_nl1.shape)

        with gene_plate as ind_g:
            # assert eps_g.shape == torch.Size([g]), "Got {0}".format(eps_g.shape)
            # assert alpha0_n1g.shape == torch.Size([len(ind_n), 1, g]), "Got {0}".format(alpha0_n1g.shape)
            # assert alpha_nlg.shape == torch.Size([len(ind_n), covariates, len(ind_g)]), "Got {0}".format(
            #    alpha_nlg.shape)

            log_mu_n1g = alpha0_n1g[..., ind_g] + torch.sum(covariate_sub_nl1 * alpha_nlg, dim=-2, keepdim=True)
            eps_sub_g = eps_g[ind_g]

            pyro.sample("counts",
                        LogNormalPoisson(n_trials=total_umi_n11,
                                         log_rate=log_mu_n1g,
                                         noise_scale=eps_sub_g,
                                         num_quad_points=8),
                        obs=counts_ng[ind_n, None].index_select(dim=-1, index=ind_g) if observed else None)


# This guide does MAP. Everything is a parameter except eps_n1g
def guide_poisson_log_normal(
        dataset: GeneDataset,
        regularize_alpha: bool = True,
        use_covariates: bool = True,
        subsample_size_cells: int = None,
        subsample_size_genes: int = None,
        **kargs):

    # Unpack the dataset
    n, g = dataset.counts.shape[:2]
    n, covariates = dataset.covariates.shape[:2]
    cell_type_ids_n = dataset.cell_type_ids.long()  # ids: 0,1,...,K-1
    k = dataset.k_cell_types

    if torch.cuda.is_available():
        cell_type_ids_n = cell_type_ids_n.cuda()
    device = cell_type_ids_n.device

    # Define the gene and cell plates. It make sense to subsample only gene and cells.
    # cell_plate = pyro.plate("cells", size=n, dim=-3, device=device, subsample_size=subsample_size_cells)
    cell_types_plate = pyro.plate("cell_types", size=k, dim=-3, device=device)
    covariate_plate = pyro.plate("covariate", size=covariates, dim=-2, device=device)
    gene_plate = pyro.plate("genes", size=g, dim=-1, device=device, subsample_size=subsample_size_genes)

    alpha_param_loc_klg = pyro.param("alpha_loc", torch.zeros((k, covariates, g), device=device))

    with gene_plate as ind_g:
        with cell_types_plate:
            with covariate_plate:
                if use_covariates:
                    alpha_loc_tmp = alpha_param_loc_klg[..., ind_g]
                else:
                    alpha_loc_tmp = torch.zeros_like(alpha_param_loc_klg[..., ind_g])
                alpha_klg = pyro.sample("alpha", dist.Delta(v=alpha_loc_tmp))
                assert alpha_klg.shape == torch.Size([k, covariates, len(ind_g)])


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