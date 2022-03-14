from typing import Tuple, Optional, Union, List
import torch
import numpy
import pyro
import time
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
import matplotlib.pyplot as plt
from .gene_utils import GeneDataset


class LogNormalPoisson(TorchDistribution):
    """
    A Poisson distribution with rate: :math:`r = N \times \\exp\\left[ \\log \\mu + \\epsilon \\right]`
    where noise is normally distributed with mean zero and variance sigma, i.e. :math:`\\epsilon \\sim N(0, \\sigma)`.

    See `Mingyuan <http://people.ee.duke.edu/~lcarin/Mingyuan_ICML_2012.pdf>`_ for discussion
    of the nice properties of the LogNormalPoisson model.
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
            n_trials: non-negative number of Poisson trials, i.e. `N`.
            log_rate: the log_rate of a single trial, i.e. :math:`\\log \\mu`.
            noise_scale: controls the level of the injected noise, i.e. :math:`\\sigma`.
            num_quad_points: number of quadrature points used to compute the (approximate) `log_prob`. Defaults to 8.
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
    Given the cell-type labels and some covariates the model predicts the gene expression.
    The counts are modelled as a LogNormalPoisson process. See documentation for more details.
    """

    def __init__(self):
        self._optimizer = None
        self._optimizer_initial_state = None
        self._loss_history = []
        self._train_kargs = None

    def _get_gene_list(self) -> List[str]:
        return self._train_kargs["gene_names"]

    def _get_cell_type_mapping(self) -> dict:
        return self._train_kargs["cell_type_mapping"]

    def _get_inverse_cell_type_mapping(self) -> dict:
        cell_type_mapping = self._get_cell_type_mapping()

        # Invert the cell_type_mapping (of the form: "cell_type" -> integer code)
        # to inverse_cell_type_mapping (of the form: integer_code -> "cell_types")
        # Note that multiple cell_types can be assigned to the same integer codes thefefor the inversion need
        # to keep track of possible name collisions
        inverse_cell_type_mapping = dict()
        for cell_type_name, code in cell_type_mapping.items():
            try:
                existing = inverse_cell_type_mapping[code]
                inverse_cell_type_mapping[code] = existing + "_AND_" + str(cell_type_name)
            except KeyError:
                inverse_cell_type_mapping[code] = str(cell_type_name)
        return inverse_cell_type_mapping

    def _get_cell_type_names_kg(self, g) -> numpy.ndarray:
        """ Return a numpy.array of shape k_cell_type by g with the cell_type_names """
        inverse_cell_type_mapping = self._get_inverse_cell_type_mapping()
        k_cell_types = len(inverse_cell_type_mapping.keys())
        cell_types_codes = torch.arange(k_cell_types).view(-1, 1).expand(k_cell_types, g)
        cell_types_names_kg = numpy.array(list(inverse_cell_type_mapping.values()))[cell_types_codes.cpu().numpy()]
        return cell_types_names_kg

    def _get_gene_names_kg(self, k: int) -> numpy.ndarray:
        """ Return a numpy.array of shape k by genes with the gene_names """
        gene_names_list = self._get_gene_list()
        len_genes = len(gene_names_list)
        gene_codes = torch.arange(len_genes).view(1, -1).expand(k, len_genes)
        gene_names_kg = numpy.array(gene_names_list)[gene_codes.cpu().numpy()]
        return gene_names_kg

    def _model(self,
               n_cells: int,
               g_genes: int,
               l_cov: int,
               k_cell_types: int,
               use_covariates: bool,
               counts_ng: torch.Tensor,
               total_umi_n: torch.Tensor,
               covariates_nl: torch.Tensor,
               cell_type_ids_n: torch.Tensor,
               eps_range: Tuple[float, float],
               l1_regularization_strength: float,
               l2_regularization_strength: float,
               subsample_size_cells: int,
               subsample_size_genes: int,
               **kargs):

        # Define the right device:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        one = torch.ones(1, device=device)
        zero = torch.zeros(1, device=device)

        # Define the plates (i.e. conditional independence). It make sense to subsample only gene and cells.
        cell_plate = pyro.plate("cells", size=n_cells, dim=-3, device=device, subsample_size=subsample_size_cells)
        cell_types_plate = pyro.plate("cell_types", size=k_cell_types, dim=-3, device=device)
        covariate_plate = pyro.plate("covariate", size=l_cov, dim=-2, device=device)
        gene_plate = pyro.plate("genes", size=g_genes, dim=-1, device=device, subsample_size=subsample_size_genes)

        eps_k1g = pyro.param("eps",
                             0.5 * (eps_range[0] + eps_range[1]) * torch.ones((k_cell_types, 1, g_genes),
                                                                              device=device),
                             constraint=constraints.interval(lower_bound=eps_range[0],
                                                             upper_bound=eps_range[1]))
        beta0_k1g = pyro.param("beta0", torch.zeros((k_cell_types, 1, g_genes), device=device))

        with gene_plate:
            with cell_types_plate:
                with covariate_plate:
                    if l1_regularization_strength is not None:
                        # l1 prior
                        beta_klg = pyro.sample("beta_cov", dist.Laplace(loc=0, scale=one / l1_regularization_strength))
                    elif l2_regularization_strength is not None:
                        # l2 prior
                        beta_klg = pyro.sample("beta_cov", dist.Normal(loc=zero, scale=one / l2_regularization_strength))
                    else:
                        # flat prior
                        beta_klg = pyro.sample("beta_cov", dist.Uniform(low=-2*one, high=2*one))

        with cell_plate as ind_n:
            cell_ids_sub_n = cell_type_ids_n[ind_n].to(device)
            beta0_n1g = beta0_k1g[cell_ids_sub_n]
            eps_n1g = eps_k1g[cell_ids_sub_n]
            beta_nlg = beta_klg[cell_ids_sub_n]
            total_umi_n11 = total_umi_n[ind_n, None, None].to(device)
            if use_covariates:
                covariate_sub_nl1 = covariates_nl[cell_ids_sub_n].unsqueeze(dim=-1).to(device)

            with gene_plate as ind_g:
                eps_sub_n1g = eps_n1g[..., ind_g]
                if use_covariates:
                    log_mu_n1g = beta0_n1g[..., ind_g] + torch.sum(covariate_sub_nl1 * beta_nlg, dim=-2, keepdim=True)
                else:
                    log_mu_n1g = beta0_n1g[..., ind_g]

                pyro.sample("counts",
                            LogNormalPoisson(n_trials=total_umi_n11,
                                             log_rate=log_mu_n1g,
                                             noise_scale=eps_sub_n1g,
                                             num_quad_points=8),
                            obs=counts_ng[ind_n.cpu(), None].index_select(dim=-1, index=ind_g.cpu()).to(device))

    def _guide(self,
               g_genes: int,
               l_cov: int,
               k_cell_types: int,
               use_covariates: bool,
               subsample_size_genes: int,
               **kargs):

        # Define the right device:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Define the gene and cell plates. It make sense to subsample only gene and cells.
        # cell_plate = pyro.plate("cells", size=n, dim=-3, device=device, subsample_size=subsample_size_cells)
        cell_types_plate = pyro.plate("cell_types", size=k_cell_types, dim=-3, device=device)
        covariate_plate = pyro.plate("covariate", size=l_cov, dim=-2, device=device)
        gene_plate = pyro.plate("genes", size=g_genes, dim=-1, device=device, subsample_size=subsample_size_genes)

        beta_param_loc_klg = pyro.param("beta", torch.zeros((k_cell_types, l_cov, g_genes), device=device))

        with gene_plate as ind_g:
            with cell_types_plate:
                with covariate_plate:
                    if use_covariates:
                        beta_loc_tmp = beta_param_loc_klg[..., ind_g]
                    else:
                        beta_loc_tmp = torch.zeros_like(beta_param_loc_klg[..., ind_g])
                    beta_klg = pyro.sample("beta_cov", dist.Delta(v=beta_loc_tmp))
                    # assert beta_klg.shape == torch.Size([k_cell_types, l_cov, len(ind_g)])

    @property
    def optimizer(self) -> pyro.optim.PyroOptim:
        """ The optimizer associated with this model. """
        assert self._optimizer is not None, "Optimizer is not specified. Call configure_optimizer first."
        return self._optimizer

    def configure_optimizer(self, optimizer_type: str = 'adam', lr: float = 5E-3):
        """ Configure the optimizer to use. For now only adam is implemented. """
        if optimizer_type == 'adam':
            self._optimizer = pyro.optim.Adam({"lr": lr})
        else:
            raise NotImplementedError

        self._optimizer_initial_state = self._optimizer.get_state()

    def save_ckpt(self, filename: str):
        """
        Save the full state of the model and optimizer to disk.
        Use it in pair with :meth:`load_ckpt`.

        Note:
            Pyro saves unconstrained parameters and the constrain transformation.
            This means that if you manually "look inside" the ckpt you will see strange values.
            To get the actual value of the fitted parameter use the :meth:`get_params` method.
        """
        ckpt = {
            "param_store": pyro.get_param_store().get_state(),
            "optimizer": self._optimizer,
            "optimizer_state": self._optimizer.get_state(),
            "optimizer_initial_state": self._optimizer_initial_state,
            "loss_history": self._loss_history,
            "train_kargs": self._train_kargs
        }

        with open(filename, "wb") as output_file:
            torch.save(ckpt, output_file)

    def load_ckpt(self, filename: str, map_location=None):
        """
        Load the full state of the model and optimizer from disk.
        Use it in pair with :meth:`save_ckpt`.
        """

        with open(filename, "rb") as input_file:
            ckpt = torch.load(input_file, map_location)

        pyro.clear_param_store()
        pyro.get_param_store().set_state(ckpt["param_store"])
        self._optimizer = ckpt["optimizer"]
        self._optimizer.set_state(ckpt["optimizer_state"])
        self._optimizer_initial_state = ckpt["optimizer_initial_state"]
        self._loss_history = ckpt["loss_history"]
        self._train_kargs = ckpt["train_kargs"]

    def get_params(self) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Returns:
            df: dataframe with the fitted parameters.

        Note:
            This method can be used in combination with :math:`load_ckpt` to inspect
            the fitted parameters of a previous run.

        Examples:
            >>> gr = GeneRegression()
            >>> gr.load_ckpt(filename="my_old_ckpt.pt")
            >>> df_beta0, df_beta, df_eps = gr.get_params()
            >>> df_beta0.head()
        """

        # get all the fitted parameters
        mydict = dict()
        for k, v in pyro.get_param_store().items():
            mydict[k] = v.detach().cpu()

        assert set(mydict.keys()) == {"beta0", "beta", "eps"}, \
            "Error. Unexpected parameter names {}".format(mydict.keys())

        k_cell_types = mydict["beta"].shape[0]
        len_genes = mydict["beta"].shape[-1]
        cell_types_names_kg = self._get_cell_type_names_kg(g=len_genes)
        gene_names_kg = self._get_gene_names_kg(k=k_cell_types)

        # check shapes
        # eps.shape = (cell_type, 1, genes)
        assert mydict["eps"].shape == torch.Size([k_cell_types, 1, len_genes]), \
            "Unexpected shape for eps {}".format(mydict["eps"].shape)

        # beta0.shape = (cell_types, 1, genes)
        assert mydict["beta0"].shape == torch.Size([k_cell_types, 1, len_genes]), \
            "Unexpected shape for beta0 {}".format(mydict["beta0"].shape)

        # beta.shape = (cell_types, covariates, genes)
        tmp_a, tmp_b, tmp_c = mydict["beta"].shape
        assert tmp_a == k_cell_types and tmp_c == len_genes, \
            "Unexpected shape for beta {}".format(mydict["beta"].shape)

        # Create dataframe
        beta = mydict["beta"].permute(0, 2, 1)  # shape: (cell_types, genes, covariates)
        columns = ["beta_{}".format(i+1) for i in range(beta.shape[-1])]
        df = pd.DataFrame(beta.flatten(end_dim=-2).cpu().numpy(), columns=columns)
        df["beta_0"] = mydict["beta0"].squeeze(dim=-2).flatten().cpu().numpy()
        df["eps"] = mydict["eps"].squeeze(dim=-2).flatten().cpu().numpy()
        df["cell_type"] = cell_types_names_kg.flatten()
        df["gene"] = gene_names_kg.flatten()
        return df

    def show_loss(self, figsize: Tuple[float, float] = (4, 4), logx: bool = False, logy: bool = False, ax=None):
        """
        Show the loss history. Useful for checking if the training has converged.

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
              l1_regularization_strength: float = 0.1,
              l2_regularization_strength: float = None,
              eps_range: Tuple[float, float] = (1.0E-3, 1.0),
              subsample_size_cells: int = None,
              subsample_size_genes: int = None,
              from_scratch: bool = True,
              **kargs
              ):
        """
        Train the model. The trained parameter are stored in the pyro.param_store and
        can be accessed via :meth:`get_params`.

        Args:
            dataset: Dataset to train the model on
            n_steps: number of training step
            print_frequency: how frequently to print loss to screen
            use_covariates: if true, use covariates, if false use cell type information only
            l1_regularization_strength: controls the strength of the L1 regularization on the regression coefficients.
                If None there is no L1 regularization.
            l2_regularization_strength: controls the strength of the L2 regularization on the regression coefficients.
                If None there is no L2 regularization.
            eps_range: range of the possible values of the gene-specific noise. Must the a strictly positive range.
            subsample_size_genes: for large dataset, the minibatch can be created using a subset of genes.
            subsample_size_cells: for large dataset, the minibatch can be created using a subset of cells.
            from_scratch: it True (defaults) the training starts from scratch. If False the training continues
                from where it was left off. Useful for extending a previously started training.
            kargs: unused parameters

        Note:
            If you get an out-of-memory error try to tune the :attr:`subsample_size_cells`
            and :attr:`subsample_size_genes`.
        """

        if from_scratch:
            pyro.clear_param_store()
            self._loss_history = []
            assert self.optimizer is not None, "Optimizer is not specified. Call configure_optimizer first."
            self.optimizer.set_state(self._optimizer_initial_state)
        steps_completed = len(self._loss_history)

        # check validity
        assert l1_regularization_strength is None or l1_regularization_strength > 0.0
        assert l2_regularization_strength is None or l2_regularization_strength > 0.0
        assert not (l1_regularization_strength is not None and l2_regularization_strength is not None), \
            "You can NOT define both l1_regularization_strength and l2_regularization_strength."
        assert eps_range[1] > eps_range[0] > 0

        # prepare train kargs dict
        train_kargs = {
            'use_covariates': use_covariates,
            'l1_regularization_strength': l1_regularization_strength,
            'l2_regularization_strength': l2_regularization_strength,
            'eps_range': eps_range,
            'subsample_size_genes': subsample_size_genes,
            'subsample_size_cells': subsample_size_cells,
            'cell_type_mapping': dataset.cell_type_mapping,
            'gene_names': dataset.gene_names,
        }

        # make a copy so that can edit train_kargs without changing _train_kargs
        self._train_kargs = train_kargs.copy()

        # Unpack the dataset and run the SVI
        counts_ng = dataset.counts.long()
        train_kargs["n_cells"] = counts_ng.shape[0]
        train_kargs["g_genes"] = counts_ng.shape[1]
        train_kargs["l_cov"] = dataset.covariates.shape[-1]
        train_kargs["k_cell_types"] = dataset.k_cell_types
        train_kargs["counts_ng"] = counts_ng.cpu()
        train_kargs["total_umi_n"] = counts_ng.sum(dim=-1).cpu()
        train_kargs["covariates_nl"] = dataset.covariates.float().cpu()
        train_kargs["cell_type_ids_n"] = dataset.cell_type_ids.long()

        start_time = time.time()
        svi = SVI(self._model, self._guide, self.optimizer, loss=Trace_ELBO())
        for i in range(steps_completed+1, steps_completed + n_steps + 1):
            loss = svi.step(**train_kargs)
            self._loss_history.append(loss)
            if (i % print_frequency == 0) or (i == steps_completed+1):
                print('[iter {}]  loss: {:.4f}'.format(i, loss))
        print("Training completed in {} seconds".format(time.time()-start_time))

    @torch.no_grad()
    def predict(self,
                dataset: GeneDataset,
                num_samples: int = 10,
                subsample_size_cells: int = None,
                subsample_size_genes: int = None,
                ) -> (pd.DataFrame, pd.DataFrame):
        """
        Use the parameters currently in the param_store to run the prediction.
        If you want to run the prediction based on a different set of parameters you need
        to call :meth:`load_ckpt` first.

        Args:
            dataset: the dataset to run the prediction on
            num_samples: how many random samples to draw from the predictive distribution
            subsample_size_cells: if not None (defaults) the prediction are made in chunks to avoid memory issue
            subsample_size_genes: if not None (defaults) the prediction are made in chunks to avoid memory issue

        Returns:
            df_metric: For each cell_type and gene we report few metrics:
                :math:`Q_\\text{data} = E\\left[|X_{i,g} - X_{j,g}|\\right]`,
                where :math:`X_i` and :math:`X_j` are two different cells (of the same type)
                :math:`Q_\\text{pred} = E\\left[|X_{i,g} - Y_{i,g}|\\right]`,
                where `X` is the (observed) data and `Y` is a sample from the predicted posterior,
                :math:`\\text{log_score} = \\log p_\\text{posterior}\\left(X_\\text{data}\\right)`
            df_counts: For each cell and gene we report the observed counts and a sampple from the posterior
        """

        n, g = dataset.counts.shape[:2]
        n, l = dataset.covariates.shape[:2]
        k = dataset.k_cell_types

        # params
        eps_k1g = pyro.get_param_store().get_param("eps").float().cpu()
        beta0_k1g = pyro.get_param_store().get_param("beta0").float().cpu()
        beta_klg = pyro.get_param_store().get_param("beta").float().cpu()

        # dataset
        counts_ng = dataset.counts.long().cpu()
        cell_type_ids = dataset.cell_type_ids.long().cpu()
        covariates_nl1 = dataset.covariates.unsqueeze(dim=-1).float().cpu()

        assert eps_k1g.shape == torch.Size([k, 1, g]), \
            "Got {0}. Are you predicting on the right dataset?".format(eps_k1g.shape)
        assert beta0_k1g.shape == torch.Size([k, 1, g]), \
            "Got {0}. Are you predicting on the right dataset?".format(beta0_k1g.shape)
        assert beta_klg.shape == torch.Size([k, l, g]), \
            "Got {0}. Are you predicting on the right dataset?".format(beta_klg.shape)

        # prepare storage
        device_calculation = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        q_ng = torch.zeros((n, g), dtype=torch.float, device=torch.device("cpu"))
        pred_counts_ng = torch.zeros((n, g), dtype=torch.long, device=torch.device("cpu"))
        log_score_ng = torch.zeros((n, g), dtype=torch.float, device=torch.device("cpu"))

        # Loop to fill the predictions for all cell and genes
        subsample_size_cells = n if subsample_size_cells is None else subsample_size_cells
        subsample_size_genes = g if subsample_size_genes is None else subsample_size_genes

        for n_left in range(0, n, subsample_size_cells):
            n_right = min(n_left + subsample_size_cells, n)

            subn_cell_ids = cell_type_ids[n_left:n_right]
            subn_counts_ng = counts_ng[n_left:n_right]
            subn_covariates_nl1 = covariates_nl1[n_left:n_right]
            subn_total_umi_n1 = subn_counts_ng.sum(dim=-1, keepdim=True)

            for g_left in range(0, g, subsample_size_genes):
                g_right = min(g_left + subsample_size_genes, g)

                eps_n1g = eps_k1g[..., g_left:g_right][subn_cell_ids]
                beta0_n1g = beta0_k1g[..., g_left:g_right][subn_cell_ids]
                log_rate_n1g = beta0_n1g + torch.sum(subn_covariates_nl1 *
                                                     beta_klg[..., g_left:g_right][subn_cell_ids],
                                                     dim=-2, keepdim=True)

                assert subn_total_umi_n1.shape == torch.Size([n_right-n_left, 1])
                assert log_rate_n1g.shape == torch.Size([n_right-n_left, 1, g_right-g_left])
                assert beta0_n1g.shape == torch.Size([n_right-n_left, 1, g_right-g_left])
                assert eps_n1g.shape == torch.Size([n_right-n_left, 1, g_right-g_left])

                mydist = LogNormalPoisson(
                    n_trials=subn_total_umi_n1.to(device_calculation),
                    log_rate=log_rate_n1g.squeeze(dim=-2).to(device_calculation),
                    noise_scale=eps_n1g.squeeze(dim=-2).to(device_calculation),
                    num_quad_points=8)

                subn_subg_counts_ng = subn_counts_ng[..., g_left:g_right].to(device_calculation)
                log_score_ng[n_left:n_right, g_left:g_right] = mydist.log_prob(subn_subg_counts_ng).cpu()

                # compute the Q metric, i.e. |x_obs - x_pred| averaged over the multiple posterior samples
                pred_counts_tmp_bng = mydist.sample(sample_shape=torch.Size([num_samples]))
                q_ng_tmp = (pred_counts_tmp_bng - subn_subg_counts_ng).abs().float().mean(dim=-3)
                q_ng[n_left:n_right, g_left:g_right] = q_ng_tmp.cpu()
                pred_counts_ng[n_left:n_right, g_left:g_right] = pred_counts_tmp_bng[0].long().cpu()

        # average by cell_type to obtain q_prediction
        unique_cell_types = torch.unique(cell_type_ids)
        q_kg = torch.zeros((unique_cell_types.shape[0], g), dtype=torch.float, device=torch.device("cpu"))
        log_score_kg = torch.zeros((unique_cell_types.shape[0], g), dtype=torch.float, device=torch.device("cpu"))
        for k, cell_type in enumerate(unique_cell_types):
            mask = (cell_type_ids == cell_type)
            log_score_kg[k] = log_score_ng[mask].mean(dim=0)
            q_kg[k] = q_ng[mask].mean(dim=0)

        # calculate_q_data
        q_data_kg = self._calculate_q_data(cell_type_ids, counts_ng, n_pairs=num_samples)

        # Compute df_metric_kg
        # combine: gene_names_kg, cell_types_names_kg, q_kg, q_data_kg, log_score_kg into a dataframe
        cell_types_names_kg = self._get_cell_type_names_kg(g=len(dataset.gene_names))
        k_cell_types, len_genes = cell_types_names_kg.shape
        gene_names_kg = self._get_gene_names_kg(k=k_cell_types)
        assert gene_names_kg.shape == cell_types_names_kg.shape == q_kg.shape == q_data_kg.shape == log_score_kg.shape, \
            "Shape mismatch {0} vs {1} vs {2} vs {3}".format(gene_names_kg.shape,
                                                             cell_types_names_kg.shape,
                                                             q_kg.shape,
                                                             q_data_kg.shape,
                                                             log_score_kg.shape)

        df_metric_kg = pd.DataFrame(cell_types_names_kg.flatten(), columns=["cell_type"])
        df_metric_kg["gene"] = gene_names_kg.flatten()
        df_metric_kg["q_predictive"] = q_kg.flatten().cpu().numpy()
        df_metric_kg["q_empirical"] = q_data_kg.flatten().cpu().numpy()
        df_metric_kg["log_score"] = log_score_kg.flatten().cpu().numpy()

        # Compute df_counts_ng
        cell_type_ids_ng = cell_type_ids.view(-1, 1).expand(n, g)
        cell_names_ng = numpy.array(list(self._get_inverse_cell_type_mapping().values()))[cell_type_ids_ng.cpu().numpy()]
        self._get_gene_names_kg(k=cell_names_ng.shape[0])
        assert cell_names_ng.shape == dataset.counts.shape == pred_counts_ng.shape == cell_names_ng.shape, \
            "Shape mismatch {0} vs {1} vs {2} vs {3}".format(cell_names_ng.shape,
                                                             dataset.counts.shape,
                                                             pred_counts_ng.shape,
                                                             cell_names_ng.shape)

        df_counts_ng = pd.DataFrame(pred_counts_ng.flatten().cpu().numpy(), columns=["counts_pred"])
        df_counts_ng["counts_obs"] = dataset.counts.flatten().cpu().numpy()
        df_counts_ng["cell_type"] = cell_names_ng.flatten()
        df_counts_ng["gene"] = gene_names_kg.flatten()

        # return
        return df_metric_kg, df_counts_ng

    def extend_train(
            self,
            dataset: GeneDataset,
            n_steps: int = 2500,
            print_frequency: int = 50):
        """
        Utility methods which calls :meth:`train` with the same parameter just used effectively extending the training.

        Args:
            dataset: Dataset to train the model on
            n_steps: number of training step
            print_frequency: how frequently to print loss to screen
        """

        self.train(
            dataset=dataset,
            n_steps=n_steps,
            print_frequency=print_frequency,
            from_scratch=False,
            **self._train_kargs)

    def train_and_test(
            self,
            train_dataset: GeneDataset,
            test_dataset: GeneDataset,
            test_num_samples: int = 10,
            train_steps: int = 2500,
            train_print_frequency: int = 50,
            use_covariates: bool = True,
            l1_regularization_strength: float = 0.1,
            l2_regularization_strength: float = None,
            eps_range: Tuple[float, float] = (1.0E-3, 1.0),
            subsample_size_cells: int = None,
            subsample_size_genes: int = None,
            from_scratch: bool = True) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, torch.Tensor):
        """
        Utility method which sequentially calls the methods :meth:`train` and :meth:`predict`.

        Args:
            train_dataset: Dataset to train the model on
            test_dataset: Dataset to run the prediction on
            test_num_samples: how many random samples to draw from the predictive distribution
            train_steps: number of training step
            train_print_frequency: how frequently to print loss to screen during training
            use_covariates: if true, use covariates, if false use cell type information only
            l1_regularization_strength: controls the strength of the L1 regularization on the regression coefficients.
                If None there is no L1 regularization.
            l2_regularization_strength: controls the strength of the L2 regularization on the regression coefficients.
                If None there is no L2 regularization.
            eps_range: range of the possible values of the gene-specific noise. Must the a strictly positive range.
            subsample_size_genes: for large dataset, the minibatch can be created using a subset of genes.
            subsample_size_cells: for large dataset, the minibatch can be created using a subset of cells.
            from_scratch: it True (defaults) the training starts from scratch. If False the training continues
                from where it was left off. Useful for extending a previously started training.

        Returns:
            See :meth:`predict`.
        """

        self.train(
            dataset=train_dataset,
            n_steps=train_steps,
            print_frequency=train_print_frequency,
            use_covariates=use_covariates,
            l1_regularization_strength=l1_regularization_strength,
            l2_regularization_strength=l2_regularization_strength,
            eps_range=eps_range,
            subsample_size_cells=subsample_size_cells,
            subsample_size_genes=subsample_size_genes,
            from_scratch=from_scratch)

        print("training completed")

        return self.predict(
            dataset=test_dataset,
            num_samples=test_num_samples,
            subsample_size_cells=subsample_size_cells,
            subsample_size_genes=subsample_size_genes)

    def extend_train_and_test(
            self,
            train_dataset: GeneDataset,
            test_dataset: GeneDataset,
            test_num_samples: int = 10,
            train_steps: int = 2500,
            train_print_frequency: int = 50) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, torch.Tensor):
        """
        Utility method which sequentially calls the methods :meth:`extend_train` and :meth:`predict`.

        Args:
            train_dataset: Dataset to train the model on
            test_dataset: Dataset to run the prediction on
            test_num_samples: how many random samples to draw from the predictive distribution
            train_steps: number of training step
            train_print_frequency: how frequently to print loss to screen during training

        Returns:
            See :meth:`predict`.
        """

        self.extend_train(
            dataset=train_dataset,
            n_steps=train_steps,
            print_frequency=train_print_frequency)

        print("training completed")

        return self.predict(
            dataset=test_dataset,
            num_samples=test_num_samples,
            subsample_size_cells=self._train_kargs["subsample_size_cells"],
            subsample_size_genes=self._train_kargs["subsample_size_genes"])

    @staticmethod
    @torch.no_grad()
    def _calculate_q_data(cell_type_ids: torch.Tensor, counts_ng: torch.Tensor, n_pairs: Union[int, str] = 10):
        """
        For each cell-type computes :math:`E\\left[x_{i,g} - x_{j,g}\\right]` where :math:`x_{i,g}`
        are the observed counts in cell `i` for gene `g`.

        Note that the indices `i,j`
        loop over all pair of cells belonging to the same cell-type.

        Args:
            cell_type_ids: array of shape `n` with the cell_type ids
            counts_ng: array of shape :math:`(n, g)` with the cell by gene counts
            n_pairs: If "all" compute all possible pairs (this is an expensive :math:`O\\left(N^2\\right)` operation).
                If an int, for each cell we consider :math:`n_pairs` other cells to approximate the expectation value.

        Returns:
            Array of shape :math:`(k, g)` , i.e. cell-types by genes, measuring the spread in the DATA.
        """

        def _all_pairs(_c_ng):
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            diff_g = torch.zeros(_c_ng.shape[-1], device=device, dtype=torch.float)
            for j in range(1, _c_ng.shape[0]):
                diff_g += (_c_ng[:j] - _c_ng[j]).abs().sum(dim=0)  # shape j-1
            number_of_pairs = 0.5 * _c_ng.shape[0] * (_c_ng.shape[0] - 1)
            return diff_g / number_of_pairs

        def _few_pairs(_c_ng, _n_pairs):
            # select n_pairs indices for each cell making sure not to self-select.
            shift = torch.randint(high=_c_ng.shape[0] - 1, size=(_c_ng.shape[0], _n_pairs)) + 1  # int in [1, n-1]
            base = torch.arange(_c_ng.shape[0]).view(-1, 1)
            index = (base + shift) % _c_ng.shape[0]  # shape (n, p)

            # compute the q metrics
            ref = _c_ng.unsqueeze(dim=1)  # shape (n, 1, g) memory-wise it is just a view of (n, g)
            other = _c_ng[index]          # shape (n, p, g) memory-wise it is just a view of (n, g)
            diff_g = (ref - other).abs().float().mean(dim=(0, 1))  # shape g
            return diff_g

        assert n_pairs == "all" or (isinstance(n_pairs, int) and n_pairs > 0), \
            "n_pairs must be 'all' or a positive integer. Received {0}".format(n_pairs)

        g = counts_ng.shape[-1]
        unique_cell_types = torch.unique(cell_type_ids)
        q_kg = torch.zeros((unique_cell_types.shape[0], g), dtype=torch.float, device=torch.device("cpu"))
        subsample_size_genes = min(500, g)

        for k, cell_type in enumerate(unique_cell_types):
            mask = (cell_type_ids == cell_type)
            tmp_ng = counts_ng[mask]  # select a given cell-type at the time

            for g_left in range(0, g, subsample_size_genes):
                g_right = min(g_left + subsample_size_genes, g)
                subg_tmp_ng = tmp_ng[..., g_left:g_right]  # subsample few genes
                if n_pairs == "all":
                    q_kg[k, g_left:g_right] = _all_pairs(subg_tmp_ng)
                else:
                    q_kg[k, g_left:g_right] = _few_pairs(subg_tmp_ng, n_pairs)

        return q_kg

