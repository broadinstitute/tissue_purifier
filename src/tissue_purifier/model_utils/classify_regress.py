from typing import List, Union, Callable, Tuple, Any
import torch
import numpy
from abc import ABC

from pytorch_lightning import LightningModule
from pytorch_lightning.trainer import Trainer
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.base import is_regressor, is_classifier
from tissue_purifier.misc_utils.misc import linear_warmup_and_cosine_protocol
from tissue_purifier.model_utils.beta_mixture_1d import BetaMixture1D
import pandas


def make_mlp_torch(
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        hidden_activation: torch.nn.Module,
        output_activation: torch.nn.Module):
    # assertion
    assert isinstance(input_dim, int) and input_dim >= 1, \
        "Error. Input_dim = {0} must be int >= 1.".format(input_dim)
    assert isinstance(output_dim, int) and output_dim >= 1, \
        "Error. Output_dim = {0} must be int >= 1.".format(output_dim)
    assert hidden_dims is None or isinstance(hidden_dims, List), \
        "Error. hidden_dims must a None or a List of int (possibly empty). Received {0}".format(hidden_dims)

    # architecture
    if hidden_dims is None or len(hidden_dims) == 0:
        net = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_dim, out_features=output_dim, bias=True),
            output_activation)
    else:
        modules = []
        tmp_dims = [input_dim] + hidden_dims

        # zip of 2 lists of different lengths terminates when shorter list terminates
        for dim_in, dim_out in zip(tmp_dims, tmp_dims[1:]):
            modules.append(torch.nn.Linear(in_features=dim_in, out_features=dim_out, bias=True))
            modules.append(hidden_activation)
        modules.append(torch.nn.Linear(in_features=tmp_dims[-1], out_features=output_dim, bias=True))
        modules.append(output_activation)
        net = torch.nn.Sequential(*modules)
    return net


class PlMlpBase(LightningModule):
    """ This class will be used in PlClassifier and PlRegressor (has-a relationship) """
    def __init__(
            self,
            # loss
            criterium: Callable,
            # architecture
            input_dim: int,
            output_dim: int,
            hidden_dims: List[int],
            hidden_activation: torch.nn.Module,
            output_activation: torch.nn.Module,
            # optimizer
            solver: str,
            betas: Tuple[float, float],
            momentum: float,
            alpha: float,
            # protocoll
            min_learning_rate: float,
            max_learning_rate: float,
            min_weight_decay: float,
            max_weight_decay: float,
            warm_up_epochs: int,
            warm_down_epochs: int
    ):
        super().__init__()
        # loss
        self.criterium = criterium
        self.input_dim_ = input_dim
        self.output_dim_ = output_dim

        # architecture
        self.net = make_mlp_torch(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            hidden_activation=hidden_activation,
            output_activation=output_activation
        )

        # optimizers
        self.solver = solver
        self.betas = betas
        self.momentum = momentum
        self.alpha = alpha

        # protocoll
        self.learning_rate_fn = linear_warmup_and_cosine_protocol(
            f_values=(min_learning_rate, max_learning_rate, min_learning_rate),
            x_milestones=(0, warm_up_epochs, warm_up_epochs, warm_up_epochs + warm_down_epochs))
        self.weight_decay_fn = linear_warmup_and_cosine_protocol(
            f_values=(min_weight_decay, min_weight_decay, max_weight_decay),
            x_milestones=(0, warm_up_epochs, warm_up_epochs, warm_up_epochs + warm_down_epochs))

        # hidden quantities
        self.loss_ = None
        self.loss_curve_ = []
        self.loss_accumulation_value_ = None
        self.loss_accumulation_counter_ = None

    @property
    def input_dim(self):
        return self.input_dim_

    @property
    def output_dim(self):
        return self.output_dim_

    def forward(self, x):
        y_hat = self.net(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        # Manually update the optimizer parameters
        with torch.no_grad():
            opt = self.optimizers()
            assert isinstance(opt, torch.optim.Optimizer)
            lr = self.learning_rate_fn(self.current_epoch)
            wd = self.weight_decay_fn(self.current_epoch)
            for i, param_group in enumerate(opt.param_groups):
                param_group["lr"] = lr
                param_group["weight_decay"] = wd
            # print("lr, wd", lr, wd)

        x, y, index = batch
        y_hat = self(x)
        loss = self.criterium(y_hat, y)

        # Compute few things of interests
        with torch.no_grad():
            self.loss_ = loss.detach().cpu().item()  # current value of loss
            self.loss_accumulation_value_ += self.loss_
            self.loss_accumulation_counter_ += 1

        return loss

    def on_train_epoch_start(self) -> None:
        self.loss_accumulation_value_ = 0.0
        self.loss_accumulation_counter_ = 0

    def on_train_epoch_end(self, unused=None) -> None:
        tmp = (self.loss_accumulation_value_ / self.loss_accumulation_counter_)
        self.loss_curve_.append(tmp)

    def configure_optimizers(self):
        if self.solver == 'adam':
            return torch.optim.Adam(
                self.net.parameters(),
                lr=0.0,
                weight_decay=0.0,
                betas=self.betas)
        elif self.solver == 'sgd':
            return torch.optim.SGD(
                self.net.parameters(),
                lr=0.0,
                weight_decay=0.0,
                momentum=self.momentum)
        elif self.solver == 'rmsprop':
            return torch.optim.RMSprop(
                self.net.parameters(),
                lr=0.0,
                weight_decay=0.0,
                alpha=self.alpha)


class PlNoisyBase(LightningModule):
    """ This class will be used in PlClassifier (has-a relationship).

    pytorch Lightning implementation of 'Unsupervised Label Noise Modeling and Loss Correction'
    see: https://github.com/PaulAlbert31/LabelNoiseCorrection

    New labels are computed as:
    soft_labels = (1.0-w) * labels_one_hot + w * prob_one_hot  # if hard bootstrapping
    soft_labels = (1.0-w) * labels_one_hot + w * prob          # if soft bootstrapping
    where prob is what the NN predicts, and w is the probability of label being incorrect.
    It is computed by computing the assignment probability of a 2-component Mixture Model.
    It is based on the idea that correct/incorrect labels will lead to small/large losses.

    These is an additional regularization term so that corrected labels do not collapse to a single class
    empirical_class_distribution = torch.mean(prob, dim=-2)  # mean over batch. Shape (classes)
    loss_reg = self.kl_between_class_distributions(empirical_class_distribution)

    # Loss function is sum of CE w.r.t. the corrected labels plus: lambda_reg * loss_reg
    """
    def __init__(
            self,
            # architecture
            input_dim: int,
            output_dim: int,
            hidden_dims: List[int],
            hidden_activation: torch.nn.Module,
            # optimizer
            solver: str,
            betas: Tuple[float, float],
            momentum: float,
            alpha: float,
            # loss
            bootstrap_epoch_start: int,
            lambda_reg: float,
            hard_bootstrapping: bool,
            # protocoll
            warm_up_epochs: int,
            warm_down_epochs: int,
            min_learning_rate: float,
            max_learning_rate: float,
            min_weight_decay: float,
            max_weight_decay: float,
            # loss
    ):
        """
        Args:
            input_dim: input channels
            output_dim: output channels, i.e. number of classes
            hidden_dims: size of hidden layers
            hidden_activation: activation to apply to the hidden layers
            solver: one among 'adam' or 'sgd' or 'rmsprop'
            betas: parameters for the adam optimizer (used only if :attr:'solver' is 'adam')
            momentum: parameter for sgd optimizer (used only if :attr:'solver' is 'sgd')
            alpha: parameters for the rmsprop optimizer (used only if :attr:'solver' is 'rmsprop')
            bootstrap_epoch_start: bootstrapping, i.e. correcting the labels will start after this many epochs.
                The idea is that the model learns a bit before the loss are meaningful to identify incorrect labels.
            lambda_reg: strength of regularization to avoid that the corrected labels collapse to a single class.
            hard_bootstrapping: if True the correct labels are a weighted sum of two delta-functions. If False
                the corrected labels are a weighted sum of a delta-function and the network probability.
            warm_up_epochs: epochs during which to linearly increase learning rate (at the beginning of training)
            warm_down_epochs: epochs during which to anneal learning rate with cosine protocoll (at the end of training)
            min_learning_rate: minimum learning rate (at the very beginning and end of training)
            max_learning_rate: maximum learning rate (after linear ramp)
            min_weight_decay: minimum weight decay (during the entirety of the linear ramp)
            max_weight_decay: maximum weight decay (reached at the end of training)
        """
        super().__init__()
        # loss
        self.input_dim_ = input_dim
        self.output_dim_ = output_dim

        # make a mlp
        self.net = make_mlp_torch(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            hidden_activation=hidden_activation,
            output_activation=torch.nn.Identity(),
        )

        # loss
        self.bootstrap_epoch_start = bootstrap_epoch_start
        self.lambda_reg = lambda_reg
        self.hard_bootstrapping = hard_bootstrapping

        # optimizers
        self.solver = solver
        self.betas = betas
        self.momentum = momentum
        self.alpha = alpha

        # protocol
        self.learning_rate_fn = linear_warmup_and_cosine_protocol(
            f_values=(min_learning_rate, max_learning_rate, min_learning_rate),
            x_milestones=(0, warm_up_epochs, warm_up_epochs, warm_up_epochs + warm_down_epochs))
        self.weight_decay_fn = linear_warmup_and_cosine_protocol(
            f_values=(min_weight_decay, min_weight_decay, max_weight_decay),
            x_milestones=(0, warm_up_epochs, warm_up_epochs, warm_up_epochs + warm_down_epochs))

        # BetaMixture business
        self.bmm_model = BetaMixture1D(max_iters=50)
        self.bmm_model_maxLoss = None
        self.bmm_model_minLoss = None
        self.track_loss_hard_ce = None

        # hidden quantities
        self.bmm_fitted_alpha0_curve_ = []
        self.bmm_fitted_alpha1_curve_ = []
        self.bmm_fitted_beta0_curve_ = []
        self.bmm_fitted_beta1_curve_ = []
        self.bmm_fitted_mean0_curve_ = []
        self.bmm_fitted_mean1_curve_ = []
        self.learning_rate_curve_ = []
        self.weight_decay_curve_ = []

        # for compatibility with Scikit-learn, keep track of loss and loss_curve
        self.loss_ = None
        self.loss_curve_ = []
        self.loss_accumulation_value_ = None
        self.loss_accumulation_counter_ = None

    @property
    def input_dim(self):
        return self.input_dim_

    @property
    def output_dim(self):
        return self.output_dim_

    def forward(self, x):
        # this is what will be called when asking for a prediction
        return self.net(x)

    def compute_assignment_prob_bmm_model(self, log_prob, soft_targets):
        assert log_prob.shape == soft_targets.shape  # shape (N, C)
        batch_losses = - (soft_targets * log_prob).sum(dim=-1)  # shape N
        batch_losses = (batch_losses - self.bmm_model_minLoss) / (self.bmm_model_maxLoss - self.bmm_model_minLoss)
        batch_losses.clamp_(min=1E-4, max=1.0-1E-4)
        assignment_prob = self.bmm_model.look_lookup(batch_losses)  # shape N with values in (0, 1)
        return torch.Tensor(assignment_prob, device=log_prob.device).float()  # shape N

    @staticmethod
    def kl_between_class_distributions(empirical_hist):
        assert len(empirical_hist.shape) == 1  # shape (Classes)
        prior_hist = torch.ones_like(empirical_hist)
        prior = prior_hist / prior_hist.sum()
        post = empirical_hist
        tmp = prior * (prior.log() - post.log())
        kl = torch.where(torch.isfinite(tmp), tmp, torch.zeros_like(tmp))
        return kl.sum(dim=-1)  # shape -> scalar

    @staticmethod
    def soft_ce_loss(logits, soft_targets):
        assert len(logits.shape) == 2
        assert logits.shape == soft_targets.shape  # shape (N, C)
        loss = - (soft_targets * torch.nn.functional.log_softmax(logits, dim=-1)).sum(dim=-1)  # shape N
        assert loss.shape == torch.Size([logits.shape[0]])
        return loss  # shape (N)

    @staticmethod
    def hard_ce_loss(logits, targets):
        assert len(logits.shape) == 2  # shape (N,C)
        assert len(targets.shape) == 1  # shape (N)
        assert logits.shape[0] == targets.shape[0]
        loss_ce = torch.nn.functional.cross_entropy(logits, targets, reduction='none')  # shape (N)
        assert loss_ce.shape[0] == targets.shape[0]
        return loss_ce  # shape (N)

    def train_bootstrapping_beta(self, logits, targets, lambda_reg: float = 1.0, hard_bootstrapping: bool = False):
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
        prob = torch.nn.functional.softmax(logits, dim=-1)
        labels_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.output_dim)

        # look up the assignment probability
        with torch.no_grad():
            if self.current_epoch < 1:
                w = 0.5 * torch.ones_like(targets).float()
            else:
                w = self.compute_assignment_prob_bmm_model(log_prob, labels_one_hot)
                w.clamp_(min=1E-4, max=1.0-1E-4)
            w = w.unsqueeze(dim=-1)  # add a singleton for the class_dimension

        # compute loss w.r.t. the corrected labels
        if hard_bootstrapping:
            prob_one_hot = torch.nn.functional.one_hot(torch.argmax(prob, dim=-1), num_classes=self.output_dim)
            soft_labels = (1.0-w) * labels_one_hot + w * prob_one_hot
        else:
            soft_labels = (1.0-w) * labels_one_hot + w * prob
        loss_ce_soft = self.soft_ce_loss(logits=logits, soft_targets=soft_labels)

        # add a regularization term so that corrected labels do not collapse to a single class
        empirical_class_distribution = torch.mean(prob, dim=-2)  # mean over batch. Shape (classes)
        loss_reg = self.kl_between_class_distributions(empirical_class_distribution)

        # put all the losses together
        loss_training = (loss_ce_soft + lambda_reg * loss_reg).mean()
        loss_hard_ce = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
        return {"loss_training": loss_training, "loss_hard_ce": loss_hard_ce.detach()}

    def training_step(self, batch, batch_idx):
        # Manually update the optimizer parameters
        with torch.no_grad():
            opt = self.optimizers()
            assert isinstance(opt, torch.optim.Optimizer)
            lr = self.learning_rate_fn(self.current_epoch)
            wd = self.weight_decay_fn(self.current_epoch)
            for i, param_group in enumerate(opt.param_groups):
                param_group["lr"] = lr
                param_group["weight_decay"] = wd

        # compute the loss
        x, y, index = batch
        logits = self.net(x)
        if self.current_epoch < self.bootstrap_epoch_start:
            loss_ce_hard = self.hard_ce_loss(logits=logits, targets=y)
            loss_dict = {"loss_training": loss_ce_hard.mean(), "loss_hard_ce": loss_ce_hard.detach()}
        else:
            loss_dict = self.train_bootstrapping_beta(logits=logits,
                                                      targets=y,
                                                      lambda_reg=self.lambda_reg,
                                                      hard_bootstrapping=self.hard_bootstrapping)

        # track the training losses
        with torch.no_grad():
            self.track_loss_hard_ce.append(loss_dict["loss_hard_ce"])

            # This block compute the current loss and the average over one epoch
            self.loss_ = loss_dict["loss_training"].detach().cpu().item()  # current value of loss
            self.loss_accumulation_value_ += self.loss_
            self.loss_accumulation_counter_ += 1

        return loss_dict["loss_training"]

    def on_train_epoch_start(self) -> None:
        # stuff for compatibility with scikit learn
        self.loss_accumulation_value_ = 0.0
        self.loss_accumulation_counter_ = 0

        # Track stuff for BMM model
        self.track_loss_hard_ce = []

    def on_train_epoch_end(self, unused=None) -> None:
        # stuff for compatibility with scikit learn
        tmp = (self.loss_accumulation_value_ / self.loss_accumulation_counter_)
        self.loss_curve_.append(tmp)

        # Track stuff for BMM model
        all_losses = torch.cat(self.track_loss_hard_ce, dim=0)
        max_perc = torch.quantile(all_losses, q=0.95)
        min_perc = torch.quantile(all_losses, q=0.05)

        # disregard the outliers
        all_losses = all_losses[(all_losses <= max_perc) & (all_losses >= min_perc)]

        # update the BMM model
        self.bmm_model_minLoss = min_perc
        self.bmm_model_maxLoss = max_perc

        # rescale the losses in (eps, 1-eps)
        all_losses = (all_losses - self.bmm_model_minLoss) / (self.bmm_model_maxLoss - self.bmm_model_minLoss)
        all_losses.clamp_(min=1E-4, max=1.0-1E-4)

        # fit the bmm_model based on the rescaled loss
        self.bmm_model.fit(all_losses)
        self.bmm_model.create_lookup(1)

        # keep track of the values fitted by the BetaMixtureModel
        alphas = self.bmm_model.alphas
        betas = self.bmm_model.betas
        self.bmm_fitted_alpha0_curve_.append(alphas[0])
        self.bmm_fitted_alpha1_curve_.append(alphas[1])
        self.bmm_fitted_beta0_curve_.append(betas[0])
        self.bmm_fitted_beta1_curve_.append(betas[1])

        # rescale those values from dimensionless to dimensionfull
        delta_loss = (max_perc - min_perc).cpu().numpy()
        min_loss =  min_perc.cpu().numpy()
        fitted_means = min_loss + delta_loss * alphas / (alphas + betas)
        self.bmm_fitted_mean0_curve_.append(fitted_means[0])
        self.bmm_fitted_mean1_curve_.append(fitted_means[1])

        # keep track of learning rate and weight decay
        opt = self.optimizers()
        param_group = opt.param_groups[0]
        lr = param_group["lr"]
        wd = param_group["weight_decay"]
        self.learning_rate_curve_.append(lr)
        self.weight_decay_curve_.append(wd)

    def configure_optimizers(self):
        if self.solver == 'adam':
            return torch.optim.Adam(
                params=self.net.parameters(),
                lr=0.0,
                weight_decay=0.0,
                betas=self.betas)
        elif self.solver == 'sgd':
            return torch.optim.SGD(
                params=self.net.parameters(),
                lr=0.0,
                weight_decay=0.0,
                momentum=self.momentum)
        elif self.solver == 'rmsprop':
            return torch.optim.RMSprop(
                params=self.net.parameters(),
                lr=0.0,
                weight_decay=0.0,
                alpha=self.alpha)
        else:
            raise Exception("unrecognized Optimizer. Received {0}".format(self.solver))


class BaseEstimator(ABC):
    """ This is a ABC which implementes an interface similar to scikit-learn for classification and regression. """

    def __init__(
            self,
            # architecture
            hidden_dims: List[int] = None,
            hidden_activation: str = 'relu',
            # training
            batch_size: int = 256,
            max_iter: int = 200,
            # optimizers
            solver: str = 'adam',
            alpha: float = 0.99,
            momentum: float = 0.9,
            betas: Tuple[float, float] = (0.9, 0.999),
            # protocoll
            warm_up_epochs: int = 0,
            warm_down_epochs: int = 0,
            min_learning_rate: float = 1.0E-4,
            max_learning_rate: float = 1.0E-3,
            min_weight_decay: float = 1.0E-4,
            max_weight_decay: float = 1.0E-4,
            **kargs, ):
        super().__init__()

        assert hidden_dims is None or isinstance(hidden_dims, List), \
            "Error. hidden_dim must be None or a list of int. Received {0}".format(hidden_dims)
        self.hidden_dims = hidden_dims

        if hidden_activation == 'relu':
            self.hidden_activation = torch.nn.ReLU(inplace=True)
        elif hidden_activation == 'leaky_relu':
            self.hidden_activation = torch.nn.LeakyReLU(negative_slope=0.01, inplace=True)
        else:
            raise NotImplementedError

        # optimizer stuff
        self.solver = solver
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.betas = betas
        self.alpha = alpha
        self.momentum = momentum

        # protocoll
        self.warm_up_epochs = warm_up_epochs
        self.warm_down_epochs = warm_down_epochs if warm_down_epochs > 0 else max_iter  # defaults to cosine annealing
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.min_weight_decay = min_weight_decay
        self.max_weight_decay = max_weight_decay

        # loss
        self._pl_net = None
        self._is_fit = False

    def create_trainer(self):
        return Trainer(
            logger=False,
            num_nodes=1,  # uses a single machine possibly with many gpus,
            gpus=1 if torch.cuda.device_count() > 0 else None,
            check_val_every_n_epoch=-1,
            num_sanity_val_steps=0,
            max_epochs=self.max_iter,
            num_processes=1,
            accelerator=None)

    @property
    def pl_net(self) -> Union[PlNoisyBase, PlMlpBase]:
        assert self._pl_net is not None, "Error. You need to initialize mlp before accessing it."
        return self._pl_net

    @property
    def loss_(self):
        return None if self.pl_net is None else self.pl_net.loss_

    @property
    def loss_curve_(self):
        return None if self.pl_net is None else self.pl_net.loss_curve_

    @torch.no_grad()
    def _to_torch_tensor(self, x):
        """ Convert stuff to torch tensors. Useful for training to use pytorch, GPUs"""
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, numpy.ndarray):
            return torch.from_numpy(x)
        elif isinstance(x, list):
            return torch.Tensor(x)
        else:
            raise Exception("unexpected type in _to_torch_tensor", type(x))

    @torch.no_grad()
    def _to_numpy(self, x):
        """ Convert stuff to numpy array. Useful for labels (which might be string) and for saving results. """
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        elif isinstance(x, numpy.ndarray):
            return x
        elif isinstance(x, list):
            return numpy.array(x)
        else:
            raise Exception("unexpected type in _to_numpy", type(x))

    @torch.no_grad()
    def _make_integer_labels(
            self,
            labels,
            classes: Union[List[Any], numpy.ndarray] = None) -> (torch.Tensor, numpy.ndarray):
        """
        Returns:
            integer_labels (torch tensor array) and classes (list)
        """

        classes_np = numpy.unique(self._to_numpy(labels)) if classes is None else self._to_numpy(classes)
        assert isinstance(classes_np, numpy.ndarray) and len(classes_np.shape) == 1

        # mapping labels to int_labels
        class_to_int_dict = dict(zip(classes_np, range(classes_np.shape[0])))
        labels_np = self._to_numpy(labels)
        integer_labels_torch = torch.tensor([class_to_int_dict[label] for label in labels_np])
        return integer_labels_torch, classes_np

    @property
    def is_classifier(self) -> bool:
        raise NotImplementedError

    @property
    def is_regressor(self) -> bool:
        raise NotImplementedError

    def create_pl_net(self, input_dim, output_dim) -> Union[PlNoisyBase, PlMlpBase]:
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X) -> numpy.ndarray:
        raise NotImplementedError

    def score(self, X, y) -> float:
        raise NotImplementedError


class PlRegressor(BaseEstimator):
    """ PlRegressor is-a BaseEstimator and has-a pl_net (which is a LightningModule) """
    def __init__(self, output_activation=torch.nn.Identity(), **kargs):
        self.output_activation = output_activation
        super().__init__(**kargs)

    @property
    def is_classifier(self):
        return False

    @property
    def is_regressor(self):
        return True

    def create_pl_net(self, input_dim, output_dim):
        return PlMlpBase(
            criterium=torch.nn.MSELoss(reduction='mean'),
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=self.hidden_dims,
            hidden_activation=self.hidden_activation,
            output_activation=self.output_activation,
            # optimizer
            solver=self.solver,
            betas=self.betas,
            momentum=self.momentum,
            alpha=self.alpha,
            # protocoll
            min_learning_rate=self.min_learning_rate,
            max_learning_rate=self.max_learning_rate,
            min_weight_decay=self.min_weight_decay,
            max_weight_decay=self.max_weight_decay,
            warm_up_epochs=self.warm_up_epochs,
            warm_down_epochs=self.warm_down_epochs
        )

    def fit(self, X, y):
        X = self._to_torch_tensor(X)
        y = self._to_torch_tensor(y)
        if len(y.shape) == 1:
            y.unsqueeze_(dim=-1)
        assert X.shape[:-1] == y.shape[:-1]
        index = torch.arange(y.shape[0], dtype=torch.long, device=y.device)

        if torch.cuda.device_count():
            X = X.cuda()
            y = y.cuda()
            index = index.cuda()

        train_dataset = torch.utils.data.TensorDataset(X, y, index)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self._pl_net = self.create_pl_net(input_dim=X.shape[-1], output_dim=y.shape[-1])
        trainer = self.create_trainer()

        trainer.fit(model=self.pl_net, train_dataloaders=train_loader)
        self._is_fit = True

    @torch.no_grad()
    def predict(self, X) -> numpy.ndarray:
        assert self._is_fit, "Error. Need to run fit method before you can use the predict method"
        X = self._to_torch_tensor(X)
        assert X.shape[-1] == self.pl_net.input_dim, \
            "Dimension mistmatch {0} vs {1}".format(X.shape[1], self.pl_net.input_dim)

        if torch.cuda.device_count():
            X = X.cuda()
            pl_net_tmp = self.pl_net.cuda()
        else:
            pl_net_tmp = self.pl_net

        predictions = []
        n1, n_max = 0, X.shape[0]
        while n1 < n_max:
            n2 = min(n_max, n1 + self.batch_size)
            y_hat = pl_net_tmp(X[n1:n2])
            n1 = n2
            predictions.append(y_hat)
        return torch.cat(predictions, dim=0).squeeze(dim=-1).cpu().numpy()

    @torch.no_grad()
    def score(self, X, y):
        assert self._is_fit, "Error. Need to run fit method before you can use the score method"

        X = self._to_torch_tensor(X)
        y = self._to_torch_tensor(y)
        if len(y.shape) == 1:
            y.unsqueeze_(dim=-1)

        assert X.shape[0] == y.shape[0], "Dimension mistmatch X={0}, y={1}".format(X.shape, y.shape)
        assert X.shape[-1] == self.pl_net.input_dim, \
            "Dimension mistmatch {0} vs {1}".format(X.shape[1], self.pl_net.input_dim)
        assert y.shape[-1] == self.pl_net.output_dim, \
            "Dimension mistmatch {0} vs {1}".format(y.shape[1], self.pl_net.output_dim)

        with torch.no_grad():
            y_pred = self.predict(X)
            return r2_score(
                y_true=y.squeeze(-1).detach().cpu().numpy(),
                y_pred=y_pred)


class PlClassifier(BaseEstimator):
    """ PlRegressor is-a BaseEstimator and has-a pl_net (which is a LightningModule) """
    def __init__(
            self,
            # special parameters for the noise label situation
            noisy_labels: bool = False,
            bootstrap_epoch_start: int = 100,
            lambda_reg: float = 1.0,
            hard_bootstrapping: bool = False,
            **kargs):
        """
        Args:
            noisy_labels: whether to use classification with noisy labels algorithm
            bootstrap_epoch_start: when to start correcting the noisy labels
            lambda_reg: strength of the regularization which prevents the corrected labels from collapsing
                to a single class
            hard_bootstrapping: If true the corrected labels are weighted sum of two delta-functions.
                If false are weighted sum of one-delta and the predicted probability.
            kargs: any parameter passed to :class:'BaseEstimator' such as max_iter, solver, ...
        """

        # spacial parameters which will be used only if noisy_labels == True
        self.noisy_labels = noisy_labels
        self.bootstrap_epoch_start = bootstrap_epoch_start
        self.lambda_reg = lambda_reg
        self.hard_bootstrapping = hard_bootstrapping

        # standard parameters
        self._classes_np = None
        self.output_activation = torch.nn.Identity()  # return the raw logit
        super().__init__(**kargs)

    @property
    def is_classifier(self):
        return True

    @property
    def is_regressor(self):
        return False

    def create_mlp(self, input_dim, output_dim):
        if self.noisy_labels:
            return PlNoisyBase(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=self.hidden_dims,
                hidden_activation=self.hidden_activation,
                # optimizer
                solver=self.solver,
                betas=self.betas,
                momentum=self.momentum,
                alpha=self.alpha,
                # loss
                lambda_reg=self.lambda_reg,
                hard_bootstrapping=self.hard_bootstrapping,
                bootstrap_epoch_start=self.bootstrap_epoch_start,
                # protocoll
                warm_up_epochs=self.warm_up_epochs,
                warm_down_epochs=self.warm_down_epochs,
                min_learning_rate=self.min_learning_rate,
                max_learning_rate=self.max_learning_rate,
                min_weight_decay=self.min_weight_decay,
                max_weight_decay=self.max_weight_decay)
        else:
            return PlMlpBase(
                criterium=torch.nn.CrossEntropyLoss(reduction='mean'),
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=self.hidden_dims,
                hidden_activation=self.hidden_activation,
                output_activation=self.output_activation,
                # optimizer
                solver=self.solver,
                betas=self.betas,
                momentum=self.momentum,
                alpha=self.alpha,
                # protocoll
                warm_up_epochs=self.warm_up_epochs,
                warm_down_epochs=self.warm_down_epochs,
                min_learning_rate=self.min_learning_rate,
                max_learning_rate=self.max_learning_rate,
                min_weight_decay=self.min_weight_decay,
                max_weight_decay=self.max_weight_decay)

    def fit(self, X, y):
        X = self._to_torch_tensor(X)
        labels_torch, self._classes_np = self._make_integer_labels(y)
        self._pl_net = self.create_mlp(input_dim=X.shape[-1], output_dim=self._classes_np.shape[0])
        index = torch.arange(labels_torch.shape[0], dtype=torch.long, device=labels_torch.device)

        if torch.cuda.device_count():
            X = X.cuda()
            labels_torch = labels_torch.cuda()
            index = index.cuda()

        train_dataset = torch.utils.data.TensorDataset(X.float(), labels_torch.long(), index)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        trainer = self.create_trainer()

        trainer.fit(model=self.pl_net, train_dataloaders=train_loader)
        self._is_fit = True

    @torch.no_grad()
    def get_all_logits(self, X) -> torch.Tensor:
        raw_logit_list = []
        n1, n_max = 0, X.shape[0]

        if torch.cuda.is_available():
            pl_net_tmp = self.pl_net.cuda()
            X = X.cuda()
        else:
            pl_net_tmp = self.pl_net

        while n1 < n_max:
            n2 = min(n_max, n1 + self.batch_size)
            raw_logit = pl_net_tmp(X[n1:n2])
            n1 = n2
            raw_logit_list.append(raw_logit)
        raw_logit_all_torch = torch.cat(raw_logit_list, dim=0)
        return raw_logit_all_torch

    @torch.no_grad()
    def predict(self, X) -> numpy.ndarray:
        """ Return a list with the predictions """
        assert self._is_fit, "Error. Need to run fit method before you can use the predict method"
        X = self._to_torch_tensor(X).float()
        assert X.shape[-1] == self.pl_net.input_dim, "Dimension mistmatch"
        raw_logit_all_torch = self.get_all_logits(X)
        labels = torch.argmax(raw_logit_all_torch, dim=-1).cpu().numpy()
        return self._classes_np[labels]

    @torch.no_grad()
    def score(self, X, y) -> float:
        """ Return a numpy.array with the probbabilities for the different classes """
        assert self._is_fit, "Error. Need to run fit method before you can use the score method"
        X = self._to_torch_tensor(X)
        y_true_np = self._to_numpy(y)

        assert X.shape[0] == y_true_np.shape[0], \
            "Dimension mistmatch X={0}, labels={1}".format(X.shape, y_true_np.shape)
        assert X.shape[-1] == self.pl_net.input_dim, \
            "Dimension mistmatch {0} vs {1}".format(X.shape[1], self.mlp.input_dim)
        y_pred_np = self.predict(X)
        return accuracy_score(y_true_np, y_pred_np)

    @torch.no_grad()
    def predict_proba(self, X) -> numpy.ndarray:
        """ Return a numpy.array with the probabilities for the different classes """
        assert self._is_fit, "Error. Need to run fit method before you can use the predict_proba method"
        X = self._to_torch_tensor(X).float()
        assert X.shape[-1] == self.pl_net.input_dim, "Dimension mistmatch"
        raw_logit_all = self.get_all_logits(X)
        prob = torch.nn.functional.softmax(raw_logit_all, dim=-1)
        return prob.cpu().numpy()

    @torch.no_grad()
    def predict_log_proba(self, X) -> numpy.ndarray:
        """ Return a numpy.array with the log_prob for the different classes """
        assert self._is_fit, "Error. Need to run fit method before you can use the predict_proba method"
        X = self._to_torch_tensor(X).float()
        assert X.shape[-1] == self.pl_net.input_dim, "Dimension mistmatch"
        raw_logit_all = self.get_all_logits(X)
        prob = torch.nn.functional.log_softmax(raw_logit_all, dim=-1)
        return prob.cpu().numpy()


def classify_and_regress(
        input_dict: dict,
        feature_keys: List[str],
        regress_keys: List[str] = None,
        classify_keys: List[str] = None,
        regressor: Union[KNeighborsRegressor, RidgeCV, PlRegressor] = None,
        classifier: Union[KNeighborsClassifier, RidgeClassifierCV, PlClassifier] = None,
        n_splits: int = 5,
        n_repeats: int = 1,
        verbose: bool = False) -> [pandas.DataFrame, pandas.DataFrame]:
    """
    Train a Classifier and a Regressor to use some featutes to predict other features.

    Args:
        input_dict: dict with both the feature to use and the one to predict
        regressor: the regressor to train
        classifier: the classifier to train
        feature_keys: keys corresponding to the independent variables
        regress_keys: keys corresponding to the variables to regress
        classify_keys: keys corresponding to the variables to classify
        n_splits: int, number of splits for RepeatedKFold (regressor) or RepeatedStratifiedKFold (classifier).
            If n_splits is 5 (defaults) then train_test_split is 80% - 20%.
        n_repeats: int, number of repeats for RepeatedKFold (regressor) or RepeatedStratifiedKFold (classifier).
            The total number of trained model is n_plists * n_repeats.
        verbose: bool, if true print some intermediate statements

    Returns:
        A ddataframe. Each row is a different X,y combination with the metrics describing the quality of the
        regression/classification.
    """
    if regress_keys is not None:
        assert is_regressor(regressor) or regressor.is_regressor, "Please pass in a regressor"

    if classify_keys is not None:
        assert is_classifier(classifier) or classifier.is_classifier, "Please pass in a classifier"

    assert isinstance(n_splits, int) and isinstance(n_repeats, int) and n_splits >= 1 and n_repeats >= 1, \
        "Error. n_splits = {0} and n_repeats = {1} must be integers >= 1.".format(n_splits, n_repeats)
    assert n_splits > 1 or (n_splits == 1 and n_repeats == 1), \
        "Misconfiguration error. It does not make sense to have n_splits == 1 and n_repeats != 1"

    assert isinstance(feature_keys, list), \
        "Feature_keys need to be a list. Received {0}".format(type(feature_keys))
    assert regress_keys is None or isinstance(regress_keys, list), \
        "Regress_keys need to be a list. Received {0}".format(type(regress_keys))
    assert classify_keys is None or isinstance(classify_keys, list), \
        "Classify_keys need to be a list. Received {0}".format(type(classify_keys))

    assert set(feature_keys).issubset(input_dict.keys()), \
        "Feature keys are not present in input dictionary."
    assert regress_keys is None or set(regress_keys).issubset(set(input_dict.keys())), \
        "Regress keys are not present in input dictionary."
    assert classify_keys is None or set(classify_keys).issubset(set(input_dict.keys())), \
        "Classify keys are not present in input dictionary."

    def _manual_shuffle(_X, _y):
        assert _X.shape[0] == _y.shape[0]
        random_index = numpy.random.permutation(_y.shape[0])
        return _X[random_index], _y[random_index]

    def _preprocess_to_numpy(x, len_shape: int):
        """ convert the features into a 2D numpy tensor (n, p) and the targets into a 1D numpy tensor (n) """
        assert isinstance(len_shape, int) and (len_shape == 1 or len_shape == 2)
        if isinstance(x, torch.Tensor):
            x = x.flatten(end_dim=-len_shape)
            assert len(x.shape) == len_shape
            return x.cpu().numpy()
        elif isinstance(x, numpy.ndarray):
            assert len(x.shape) == len_shape
            return x
        elif isinstance(x, list):
            assert len_shape == 1
            return numpy.array(x)

    def _do_regression(_X, _y, x_key, y_key):
        print("regression", x_key, y_key)
        mask_x = numpy.isfinite(_X)
        mask_y = numpy.isfinite(_y)
        assert numpy.all(mask_x), "ON entry {0} is not finite. {1}".format(x_key, _X[~mask_x])
        assert numpy.all(mask_y), "ON entry {0} is not finite. {1}".format(y_key, _y[~mask_y])

        _X, _y = _manual_shuffle(_X, _y)
        _tmp_dict = {}

        if n_splits > 1:
            rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
            for train_index, test_index in rkf.split(_X, _y):
                _X_train, _X_test, _y_train, _y_test = _X[train_index], _X[test_index], _y[train_index], _y[test_index]
                regressor.fit(_X_train, _y_train)

                _tmp_dict["x_key"] = _tmp_dict.get("x_key", []) + [x_key]
                _tmp_dict["y_key"] = _tmp_dict.get("y_key", []) + [y_key]
                _tmp_dict["r2_train"] = _tmp_dict.get("r2_train", []) + [regressor.score(_X_train, _y_train)]
                _tmp_dict["r2_test"] = _tmp_dict.get("r2_test", []) + [regressor.score(_X_test, _y_test)]
            _df_tmp = pandas.DataFrame(_tmp_dict, index=numpy.arange(rkf.get_n_splits()))
        elif n_splits == 1 and n_repeats == 1:
            regressor.fit(_X, _y)
            _tmp_dict = {
                "x_key": x_key,
                "y_key": y_key,
                "r2": regressor.score(_X, _y)
            }
            _df_tmp = pandas.DataFrame(_tmp_dict, index=[0])
        else:
            raise Exception("Does not make sense to have n_splits = {0} and n_repeats = {1}".format(n_splits,
                                                                                                    n_repeats))
        return _df_tmp

    def _do_classification(_X, _y, x_key, y_key):
        print("classification", x_key, y_key)
        mask_x = numpy.isfinite(_X)
        mask_y = numpy.isfinite(_y)
        assert numpy.all(mask_x), "ON entry {0} is not finite. {1}".format(x_key, _X[~mask_x])
        assert numpy.all(mask_y), "ON entry {0} is not finite. {1}".format(y_key, _y[~mask_y])

        _X, _y = _manual_shuffle(_X, _y)
        _tmp_dict = {}

        if n_splits > 1:

            rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
            for train_index, test_index in rkf.split(_X, _y):
                _X_train, _X_test, _y_train, _y_test = _X[train_index], _X[test_index], _y[train_index], _y[test_index]
                classifier.fit(_X_train, _y_train)

                _tmp_dict["x_key"] = _tmp_dict.get("x_key", []) + [x_key]
                _tmp_dict["y_key"] = _tmp_dict.get("y_key", []) + [y_key]
                _tmp_dict["accuracy_train"] = _tmp_dict.get("accuracy_test", []) + [classifier.score(_X_train, _y_train)]
                _tmp_dict["accuracy_test"] = _tmp_dict.get("accuracy_test", []) + [classifier.score(_X_test, _y_test)]
            _df_tmp = pandas.DataFrame(_tmp_dict, index=numpy.arange(rkf.get_n_splits()))

        elif n_splits == 1 and n_repeats == 1:
            classifier.fit(_X, _y)
            _tmp_dict = {
                "x_key": x_key,
                "y_key": y_key,
                "accuracy": classifier.score(_X, _y)
            }
            _df_tmp = pandas.DataFrame(_tmp_dict, index=[0])
        else:
            raise Exception("Does not make sense to have n_splits = {0} and n_repeats = {1}".format(n_splits,
                                                                                                    n_repeats))
        return _df_tmp

    # loop over everything to make the predictions
    df = None
    for feature_key in feature_keys:
        X_all = _preprocess_to_numpy(input_dict[feature_key], len_shape=2)

        if classify_keys is not None:
            for kc in classify_keys:
                if verbose:
                    print("{0} classify {1}".format(feature_key, kc))
                y_all = _preprocess_to_numpy(input_dict[kc], len_shape=1)
                tmp_df = _do_classification(X_all, y_all, x_key=feature_key, y_key=kc)
                df = tmp_df if df is None else df.merge(tmp_df, how='outer')

        if regress_keys is not None:
            for kr in regress_keys:
                if verbose:
                    print("{0} regress {1}".format(feature_key, kr))
                y_all = _preprocess_to_numpy(input_dict[kr], len_shape=1)
                tmp_df = _do_regression(X_all, y_all, x_key=feature_key, y_key=kr)
                df = tmp_df if df is None else df.merge(tmp_df, how='outer')

    return df

