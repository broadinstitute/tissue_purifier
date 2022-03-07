from typing import List, Union, Tuple, Any
import torch
import torch.utils.data
import torch.nn.functional
import numpy
from abc import ABC
from pytorch_lightning import LightningModule
from pytorch_lightning.trainer import Trainer
from sklearn.metrics import r2_score, accuracy_score
from ._pl_clean import PlMlpClean
from ._pl_noisy import PlMlpNoisy


class BaseEstimator(ABC):
    """ This is a ABC which implements an interface similar to scikit-learn for classification and regression. """

    def __init__(
            self,
            # architecture
            hidden_dims: List[int] = None,
            hidden_activation: str = 'relu',
            # training
            batch_size: int = 256,
            # optimizers
            solver: str = 'adam',
            alpha: float = 0.99,
            momentum: float = 0.9,
            betas: Tuple[float, float] = (0.9, 0.999),
            # protocoll
            warm_up_epochs: int = 0,
            warm_down_epochs: int = 0,
            max_epochs: int = 200,
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
        self.betas = betas
        self.alpha = alpha
        self.momentum = momentum

        # protocoll
        self.max_epochs = max_epochs
        self.warm_up_epochs = warm_up_epochs
        self.warm_down_epochs = warm_down_epochs
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
            max_epochs=self.max_epochs,
            num_processes=1,
            accelerator=None)

    @property
    def pl_net(self) -> LightningModule:
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

    def create_pl_net(self, input_dim, output_dim) -> LightningModule:
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
        return PlMlpClean(
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
            max_epochs=self.max_epochs,
            warm_up_epochs=self.warm_up_epochs,
            warm_down_epochs=self.warm_down_epochs,
            min_learning_rate=self.min_learning_rate,
            max_learning_rate=self.max_learning_rate,
            min_weight_decay=self.min_weight_decay,
            max_weight_decay=self.max_weight_decay,
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
            return PlMlpNoisy(
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
                max_epochs=self.max_epochs,
                warm_up_epochs=self.warm_up_epochs,
                warm_down_epochs=self.warm_down_epochs,
                min_learning_rate=self.min_learning_rate,
                max_learning_rate=self.max_learning_rate,
                min_weight_decay=self.min_weight_decay,
                max_weight_decay=self.max_weight_decay)
        else:
            return PlMlpClean(
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
                max_epochs=self.max_epochs,
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
            "Dimension mistmatch {0} vs {1}".format(X.shape[1], self.pl_net.input_dim)
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
