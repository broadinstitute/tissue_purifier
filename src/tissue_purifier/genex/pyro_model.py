from typing import Tuple, Optional
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

    def _model(self,
               dataset: GeneDataset,
               eps_range: Tuple[float, float],
               l1_regularization_strength: float = None,
               l2_regularization_strength: float = None,
               subsample_size_cells: int = None,
               subsample_size_genes: int = None,
               **kargs):

        # check validity
        assert l1_regularization_strength is None or l1_regularization_strength > 0.0
        assert l2_regularization_strength is None or l2_regularization_strength > 0.0
        assert not (l1_regularization_strength is not None and l2_regularization_strength is not None), \
            "You can NOT define both l1_regularization_strength and l2_regularization_strength."

        # Unpack the dataset
        assert eps_range[1] > eps_range[0] > 0
        counts_ng = dataset.counts.long()
        total_umi_n = counts_ng.sum(dim=-1)
        covariates_nl = dataset.covariates.float()
        cell_type_ids_n = dataset.cell_type_ids.long()  # ids: 0,1,...,K-1
        k = dataset.k_cell_types
        n, g = counts_ng.shape[:2]
        n, l_cov = covariates_nl.shape[:2]
        n = cell_type_ids_n.shape[0]

        # Define the right device:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        one = torch.ones(1, device=device)
        zero = torch.zeros(1, device=device)

        # Define the plates (i.e. conditional independence). It make sense to subsample only gene and cells.
        cell_plate = pyro.plate("cells", size=n, dim=-3, device=device, subsample_size=subsample_size_cells)
        cell_types_plate = pyro.plate("cell_types", size=k, dim=-3, device=device)
        covariate_plate = pyro.plate("covariate", size=l_cov, dim=-2, device=device)
        gene_plate = pyro.plate("genes", size=g, dim=-1, device=device, subsample_size=subsample_size_genes)

        eps_k1g = pyro.param("eps",
                             0.5 * (eps_range[0] + eps_range[1]) * torch.ones((k, 1, g), device=device),
                             constraint=constraints.interval(lower_bound=eps_range[0],
                                                             upper_bound=eps_range[1]))
        beta0_k1g = pyro.param("beta0", torch.zeros((k, 1, g), device=device))

        with gene_plate:
            with cell_types_plate:
                with covariate_plate:
                    if l1_regularization_strength is not None:
                        # l1 prior
                        beta_klg = pyro.sample("beta", dist.Laplace(loc=0, scale=one / l1_regularization_strength))
                    elif l2_regularization_strength is not None:
                        # l2 prior
                        beta_klg = pyro.sample("beta", dist.Normal(loc=zero, scale=one / l2_regularization_strength))
                    else:
                        # flat prior
                        beta_klg = pyro.sample("beta", dist.Uniform(low=-2*one, high=2*one))

        with cell_plate as ind_n:
            cell_ids_sub_n = cell_type_ids_n[ind_n].to(device)
            beta0_n1g = beta0_k1g[cell_ids_sub_n]
            eps_n1g = eps_k1g[cell_ids_sub_n]
            beta_nlg = beta_klg[cell_ids_sub_n]
            covariate_sub_nl1 = covariates_nl[cell_ids_sub_n].unsqueeze(dim=-1).to(device)
            total_umi_n11 = total_umi_n[ind_n, None, None].to(device)

            with gene_plate as ind_g:
                log_mu_n1g = beta0_n1g[..., ind_g] + torch.sum(covariate_sub_nl1 * beta_nlg, dim=-2, keepdim=True)
                eps_sub_n1g = eps_n1g[..., ind_g]

                pyro.sample("counts",
                            LogNormalPoisson(n_trials=total_umi_n11.to(device),
                                             log_rate=log_mu_n1g.to(device),
                                             noise_scale=eps_sub_n1g.to(device),
                                             num_quad_points=8),
                            obs=counts_ng[ind_n.cpu(), None].index_select(dim=-1, index=ind_g.cpu()).to(device))

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

        beta_param_loc_klg = pyro.param("beta_loc", torch.zeros((k, l, g), device=device))

        with gene_plate as ind_g:
            with cell_types_plate:
                with covariate_plate:
                    if use_covariates:
                        beta_loc_tmp = beta_param_loc_klg[..., ind_g]
                    else:
                        beta_loc_tmp = torch.zeros_like(beta_param_loc_klg[..., ind_g])
                    beta_klg = pyro.sample("beta", dist.Delta(v=beta_loc_tmp))
                    assert beta_klg.shape == torch.Size([k, l, len(ind_g)])

    def render_model(self, dataset: GeneDataset, filename: Optional[str] = None,  render_distributions: bool = False):
        """
        Wrapper around :meth:`pyro.render_model` to visualize the graphical model.

        Args:
            dataset: dataset to use for computing the shapes
            filename: File to save rendered model in. If None (defaults) the image is displayed.
            render_distributions: Whether to include RV distribution
        """
        model_kargs = {
            'dataset': dataset,
            'use_covariates': True,
            'beta_scale': None,
            'eps_range': (0.01, 0.02),
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
        """ Save the full state of the model and optimizer to disk. """
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
        """ Load the full state of the model and optimizer from disk. """

        with open(filename, "rb") as input_file:
            ckpt = torch.load(input_file, map_location)

        pyro.clear_param_store()
        pyro.get_param_store().set_state(ckpt["param_store"])
        self._optimizer = ckpt["optimizer"]
        self._optimizer.set_state(ckpt["optimizer_state"])
        self._optimizer_initial_state = ckpt["optimizer_initial_state"]
        self._loss_history = ckpt["loss_history"]

    @staticmethod
    def get_params():
        """ Returns a (detached) dictionary with fitted parameters. """
        mydict = dict()
        for k, v in pyro.get_param_store().items():
            mydict[k] = v.detach().cpu()
        return mydict

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
              from_scratch: bool = True):
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

        Note:
            If you get an out-of-memory error try to tune the :attr:`subsample_size_cells`
            and :attr:`subsample_size_genes`.
        """

        if from_scratch:
            pyro.clear_param_store()
            self._loss_history = []
            assert self.optimizer is not None, "Optimizer is not specified. Call configure_optimizer first."
            self.optimizer.set_state(self._optimizer_initial_state)

        model_kargs = {
            'dataset': dataset,
            'use_covariates': use_covariates,
            'l1_regularization_strength': l1_regularization_strength,
            'l2_regularization_strength': l2_regularization_strength,
            'eps_range': eps_range,
            'subsample_size_genes': subsample_size_genes,
            'subsample_size_cells': subsample_size_cells,
        }

        svi = SVI(self._model, self._guide, self.optimizer, loss=Trace_ELBO())
        for i in range(n_steps+1):
            loss = svi.step(**model_kargs)
            self._loss_history.append(loss)
            if i % print_frequency == 0:
                print('[iter {}]  loss: {:.4f}'.format(i, loss))

    @staticmethod
    def predict(dataset: GeneDataset, num_samples: int = 10) -> (dict, pd.DataFrame, pd.DataFrame):
        """
        Use the parameters currently in the param_store to run the prediction.
        If you want to run the prediction based on a different set of parameters you need
        to call :meth:`load_ckpt` first.

        Args:
            dataset: the dataset to run the prediction on
            num_samples: how many random samples to draw from the predictive distribution

        Returns:
            result: a dictionary with the true and predicted counts, the cell_type and two metrics (log_score and deviance)
            log_score_df: a DataFrame with the log_score evaluation metric
            deviance_df: a DataFrame with the deviance evaluation metric
        """

        n, g = dataset.counts.shape[:2]
        n, l = dataset.covariates.shape[:2]
        k = dataset.k_cell_types

        eps_k1g = pyro.get_param_store().get_param("eps")
        beta0_k1g = pyro.get_param_store().get_param("beta0")
        beta_klg = pyro.get_param_store().get_param("beta_loc")
        counts_ng = dataset.counts
        cell_type_ids = dataset.cell_type_ids.long()
        covariates_nl1 = dataset.covariates.unsqueeze(dim=-1)

        assert eps_k1g.shape == torch.Size([k, 1, g]), \
            "Got {0}. Are you predicting on the right dataset?".format(eps_k1g.shape)
        assert beta0_k1g.shape == torch.Size([k, 1, g]), \
            "Got {0}. Are you predicting on the right dataset?".format(beta0_k1g.shape)
        assert beta_klg.shape == torch.Size([k, l, g]), \
            "Got {0}. Are you predicting on the right dataset?".format(beta_klg.shape)

        if torch.cuda.is_available():
            cell_type_ids = cell_type_ids.cuda()
            eps_k1g = eps_k1g.cuda()
            beta0_k1g = beta0_k1g.cuda()
            beta_klg = beta_klg.cuda()
            counts_ng = counts_ng.cuda()
            covariates_nl1 = covariates_nl1.cuda()

        log_rate_n1g = beta0_k1g[cell_type_ids] + (covariates_nl1 * beta_klg[cell_type_ids]).sum(dim=-2, keepdim=True)
        total_umi_n1 = counts_ng.sum(dim=-1, keepdim=True)

        print("debug")
        print("total_umi_n1", total_umi_n1.shape)
        print("log_rate_n1g", log_rate_n1g.shape)
        print("eps_k1g", eps_k1g.shape)

        mydist = LogNormalPoisson(
            n_trials=total_umi_n1,
            log_rate=log_rate_n1g.squeeze(dim=-2),
            noise_scale=eps_k1g.squeeze(dim=-2),
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
            from_scratch: bool = True):
        """ Utility function which sequentially calls the methods :meth:`train` and :meth:`predict`. """

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

        return self.predict(dataset=test_dataset, num_samples=test_num_samples)
