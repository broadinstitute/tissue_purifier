import torch
import argparse
import math
import numpy
from typing import Union, Tuple
from umap.umap_ import UMAP
import leidenalg
import igraph as ig
import scipy
import torch
from torch.optim.optimizer import Optimizer


def smart_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def linear_warmup_and_cosine_protocol(
        f_values: Tuple[float, float, float],
        x_milestones: Tuple[int, int, int, int]):
    """
    There are 5 regions:
    1. constant at f0 for x < x0
    2. linear increase from f0 to f1 for x0 < x < x1
    3. constant at f1 for x1 < x < x2
    4. cosine protocol from f1 to f2 for x2 < x < x3
    5. constant at f2 for x > x3

    If you want a linear_ramp followed by a cosine_decay only simply set:
    1. x0=0 (to eliminate the first constant piece)
    2. x2=x1 (to eliminate the second constant piece)
    3. max_epochs=x3 (to make the simulation stop after the linear or cosine decay)
    """
    assert x_milestones[0] <= x_milestones[1] <= x_milestones[2] <= x_milestones[3]

    def fn(step):
        if step <= x_milestones[0]:
            return float(f_values[0])
        elif (step > x_milestones[0]) and (step <= x_milestones[1]):
            m = float(f_values[1] - f_values[0]) / float(max(1, x_milestones[1] - x_milestones[0]))
            return float(f_values[0]) + m * float(step - x_milestones[0])
        elif (step > x_milestones[1]) and (step <= x_milestones[2]):
            return float(f_values[1])
        elif (step > x_milestones[2]) and (step <= x_milestones[3]):
            progress = float(step - x_milestones[2]) / float(max(1, x_milestones[3] - x_milestones[2]))  # in (0,1)
            tmp = 0.5 * (1.0 + math.cos(math.pi * progress))  # in (1,0)
            return float(f_values[2]) + tmp * float(f_values[1] - f_values[2])
        else:
            return float(f_values[2])

    return fn


class LARS(Optimizer):
    """Extends SGD in PyTorch with LARS scaling from the paper
    `Large batch training of Convolutional Networks <https://arxiv.org/pdf/1708.03888.pdf>`_.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        trust_coefficient (float, optional): trust coefficient for computing LR (default: 0.001)
        eps (float, optional): eps for division denominator (default: 1e-8)
    Example:
        >>> model = torch.nn.Linear(10, 1)
        >>> input = torch.Tensor(10)
        >>> target = torch.Tensor([1.])
        >>> loss_fn = lambda input, target: (input - target) ** 2
        >>> #
        >>> optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    .. note::
        The application of momentum in the SGD part is modified according to
        the PyTorch standards. LARS scaling fits into the equation in the
        following fashion.
        .. math::
            \begin{aligned}
                g_{t+1} & = \text{lars_lr} * (\beta * p_{t} + g_{t+1}), \\
                v_{t+1} & = \\mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \\end{aligned}
        where :math:`p`, :math:`g`, :math:`v`, :math:`\\mu` and :math:`\beta` denote the
        parameters, gradient, velocity, momentum, and weight decay respectively.
        The :math:`lars_lr` is defined by Eq. 6 in the paper.
        The Nesterov version is analogously modified.
    .. warning::
        Parameters with weight decay set to 0 will automatically be excluded from
        layer-wise LR scaling. This is to ensure consistency with papers like SimCLR
        and BYOL.
    """

    def __init__(
        self,
        params,
        lr=None,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        trust_coefficient=0.001,
        eps=1e-8,
    ):
        if lr is None or lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            trust_coefficient=trust_coefficient,
            eps=eps,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # exclude scaling for params with 0 weight decay
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                p_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)

                # lars scaling + weight decay part
                if weight_decay != 0:
                    if p_norm != 0 and g_norm != 0:
                        lars_lr = p_norm / (g_norm + p_norm * weight_decay + group["eps"])
                        lars_lr *= group["trust_coefficient"]

                        d_p = d_p.add(p, alpha=weight_decay)
                        d_p *= lars_lr

                # sgd part
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group["lr"])

        return loss


def get_percentile(data: Union[torch.Tensor, numpy.ndarray], dim: int) -> Union[torch.Tensor, numpy.ndarray]:
    """
    Takes some data and convert it into a percentile (in [0.0, 1.0]) along a specified dimension.
    Useful to convert a tensor into the range [0.0, 1.0] for visualization.

    Args:
        data: input data to convert to percentile in [0,1].
        dim: the dimension along which to compute the quantiles

    Returns:
        percentile: torch.tensor or numpy.array (depending on the input type)
            with the same shape as the input with the percentile values.
            A percentile of 0.9 means that 90% of the input values were smaller.
    """
    assert isinstance(data, torch.Tensor) or isinstance(data, numpy.ndarray) or isinstance(data, list), \
        "Error. Input must be either a numpy.array or torch.Tensor or list. Received {0}".format(type(data))

    def _to_torch(x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, numpy.ndarray):
            return torch.from_numpy(x)

    data_torch = _to_torch(data)

    old_dims = list(data_torch.shape)
    new_dims = [1] * len(old_dims)
    new_dims[dim] = old_dims[dim]

    indices = torch.argsort(data_torch, dim=dim, descending=False)
    src = torch.linspace(0.0, 1.0, new_dims[dim]).view(new_dims).expand_as(indices).clone()
    percentile = torch.zeros_like(data_torch)
    percentile.scatter_(dim=dim, index=indices, src=src)

    if isinstance(data, numpy.ndarray):
        return percentile.cpu().numpy()
    else:
        return percentile


def inverse_one_hot(_image, bg_label: int = -1, dim: int = -3, threshold: float = 0.1):
    """
    Takes float tensor and compute the argmax and max_value along the specified dimension.
    Returns a integer tensor of the same shape as the input_tensor but with the dim removed.
    If the max_value is less than the threshold the bg_label is assigned.

    For example. It can take an image of size (ch, w, h) and generate an integer mask of size (w, h).
    This operation can be thought as the inverse of the one-hot operation which takes an integer tensor of size (*)
    and returns a float tensor with an extra dimension, for example (*, num_classes).

    Args:
        _image: any float tensor
        bg_label: integer, the value assigned to the entries of which are smaller than the threshold
        dim: int, the dimension along which to compute the max. For images this is usually the channel dimension, i.e. -3.
        threshold: float, the value of the threshold. Value smaller than this are set assigned to the background

    Returns:
        An integer mask with the same size of the input tensor but with the dim removed.

    """
    assert isinstance(bg_label, int), "Error. bg_label must be an integer. Received {0}".format(bg_label)

    _values, _indices = torch.max(_image, dim=dim)
    _mask_bg = (_values < threshold)
    _indices[_mask_bg] = bg_label
    return _indices.long()


def get_z_score(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standardize vector by removing the mean and scaling to unit variance

    Args:
        x: torch.Tensor
        dim: the dimension along which to compute the mean and std

    Return:
        The z-score, i.e. z = (x - mean) / std
    """
    std, mean = torch.std_mean(x, dim=dim, unbiased=True, keepdim=True)
    return (x-mean)/std


def compute_distance_embedding(ref_embeddings: torch.Tensor,
                               other_embeddings: torch.Tensor,
                               metric: str,
                               temperature: float = 0.5) -> torch.Tensor:
    """ Compute distance between embeddings
        Args:
            ref_embeddings: torch.Tensor of shape (*, k) where k is the dimension of the embedding
            other_embeddings: torch.Tensor of shape (n, k)
            temperature: float, the temperature used to compute contrastive distance
            metric: Can be either 'contrastive' or 'euclidean'

        Returns:
            distance of shape (*, n)
    """
    assert ref_embeddings.shape[-1] == other_embeddings.shape[-1]
    assert len(other_embeddings.shape) == 2
    assert len(ref_embeddings.shape) == 1 or len(ref_embeddings.shape) == 2

    if metric == "contrastive":
        # this is similar to the contrastive loss used to train SimClr model
        ref_embeddings_norm = torch.nn.functional.normalize(ref_embeddings, dim=-1)  # shape: (*, k)
        other_embeddings_norm = torch.nn.functional.normalize(other_embeddings, dim=-1)  # shape: (n, k)
        logits = torch.einsum('...c,nc->...n', ref_embeddings_norm, other_embeddings_norm)
        distance = torch.exp(-logits / temperature)  # Note: If similarity is high, distance is low, shape: (*, n)
        return distance

    elif metric == "euclidean":
        # ref_embeddings  # shape: (*, k)
        # other_embeddings  # shape: (n, k)
        delta = ref_embeddings.unsqueeze(dim=-2) - other_embeddings  # shape (*, n, k)
        distance2 = torch.einsum('...k, ...k -> ...', delta, delta)  # shape (*, n)
        return distance2.float().sqrt()

    elif metric == 'cosine':
        ref_embeddings_norm = torch.nn.functional.normalize(ref_embeddings, dim=-1)  # shape: (*, k)
        other_embeddings_norm = torch.nn.functional.normalize(other_embeddings, dim=-1)  # shape: (n, k)
        cosine = torch.einsum('...c,nc->...n', ref_embeddings_norm, other_embeddings_norm)
        return 1.0 + 1E-4 - cosine
    else:
        raise Exception("Wrong metric. Received {0}".format(metric))


class SmartUmap(UMAP):
    """ Return the UMAP embeddings. """

    def __init__(self,
                 preprocess_strategy: str,
                 compute_all_pairwise_distances: bool = False,
                 **kargs):
        """
        Args:
            preprocess_strategy: str, can be 'center', 'z_score', 'raw'. This is the operation to perform before UMAP
            compute_all_pairwise_distances: bool, it True (default is False) compute all pairwise distances
            **kargs: All the arguments that standard UMAP can accept
        """

        assert preprocess_strategy == 'z_score' or \
               preprocess_strategy == 'center' or \
               preprocess_strategy == 'raw', "Preprocessing \
                must be either 'center', 'z_score' or 'raw'. Received {0}".format(preprocess_strategy)

        # overwrites some UMAP default values
        kargs['n_neighbors'] = kargs.get('n_neighbors', 25)
        kargs['min_dist'] = kargs.get('min_dist', 0.5)

        # inputs
        self.preprocess_strategy = preprocess_strategy
        self.compute_all_pairwise_distances = compute_all_pairwise_distances

        # thing computed during fit
        self._distances = None
        self._mean = None
        self._std = None
        self._fitted = False
        super(SmartUmap, self).__init__(random_state=0, **kargs)

    def get_graph(self) -> scipy.sparse.coo_matrix:
        """ Returns the symmetric (sparse) matrix with the SIMILARITIES between elements """
        return self.graph_

    def get_distances(self) -> torch.Tensor:
        """ Returns the symmetric (dense) matrix with the DISTANCES between elements """
        return self._distances

    def _preprocess(self, data) -> (torch.Tensor, torch.Tensor):
        if self.preprocess_strategy == 'z_score':
            std, mean = torch.std_mean(data, dim=-2, unbiased=True, keepdim=True)
            return std, mean
        elif self.preprocess_strategy == 'center':
            mean = torch.mean(data, dim=-2, keepdim=True)
            std = torch.ones_like(mean)
            return std, mean
        elif self.preprocess_strategy == 'raw':
            mean = torch.zeros_like(data[0, :])
            std = torch.ones_like(mean)
            return std, mean

    def fit(self, data, y=None) -> "SmartUmap":
        assert y is None
        if isinstance(data, numpy.ndarray):
            data = torch.tensor(data)
        data = data.float()  # upgrade to full precision in case you are at Half

        std, mean = self._preprocess(data)
        self._mean = mean
        self._std = std
        self._fitted = True
        data = (data - mean) / std
        return super(SmartUmap, self).fit(data.detach().clone().cpu().numpy(), y)

    def transform(self, data) -> numpy.ndarray:
        """
        Use previously fitted model (including mean and std for centering and scaling the data).

        Returns:
            embeddings: numpy.tensor of shape (n_sample, n_components)
        """

        assert self._fitted, "UMAP is not fitted. Cal 'fit' or 'fit_transform' first."

        if isinstance(data, numpy.ndarray):
            data = torch.tensor(data)
        data = data.float()  # upgrade to full precision in case you are at Half

        data = (data - self._mean.to(data.device)) / self._std.to(data.device)
        embeddings = super(SmartUmap, self).transform(data.cpu().numpy())

        if self.compute_all_pairwise_distances:
            self._distances = compute_distance_embedding(
                ref_embeddings=data,
                other_embeddings=data,
                metric=self.metric).cpu().numpy()

        return embeddings

    def fit_transform(self, data, y=None) -> numpy.ndarray:
        if isinstance(data, numpy.ndarray):
            data = torch.tensor(data)
        data = data.float()  # upgrade to full precision in case you are at Half

        self.fit(data)
        return self.transform(data)


class SmartLeiden:
    def __init__(self, graph: "coo_matrix", directed: bool = True):
        """
        Args:
            graph: Usually a sparse matrix with the similarities among nodes describing the graph
            directed: bool, if True it build a directed graph.

        Note:
            The matrix obtained by the UMAP algorithm is symmetric, in that case directed should be set to True
        """
        sources, targets = graph.nonzero()
        weights = graph[sources, targets]
        if isinstance(weights, numpy.matrix):
            weights = weights.A1
        self.ig_graph = ig.Graph(directed=directed)
        self.ig_graph.add_vertices(graph.shape[0])  # this adds adjacency.shape[0] vertices
        self.ig_graph.add_edges(list(zip(sources, targets)))
        self.ig_graph.es['weight'] = weights

    def cluster(self,
                resolution: float = 1.0,
                use_weights: bool = True,
                random_state: int = 0,
                n_iterations: int = -1,
                partition_type: str = 'RBC') -> numpy.ndarray:

        partition_kwargs = {
            'resolution_parameter': resolution,
            'n_iterations': n_iterations,
            'seed': random_state,
        }

        if use_weights:
            partition_kwargs['weights'] = numpy.array(self.ig_graph.es['weight']).astype(numpy.float64)

        if partition_type == 'CPM':
            partition_type = leidenalg.CPMVertexPartition
        elif partition_type == 'RBC':
            partition_type = leidenalg.RBConfigurationVertexPartition
        else:
            raise Exception("Partitin type not recognized. Expected CPM or RBC. Received {0}".format(partition_type))

        part = leidenalg.find_partition(self.ig_graph, partition_type, **partition_kwargs)

        return numpy.array(part._membership).astype(int)


class SmartPca:
    """ Return the PCA embeddings. """

    def __init__(self,
                 preprocess_strategy: str):
        """
        Args:
            preprocess_strategy: str, can be 'center', 'z_score', 'raw'. This is the operation to perform before PCA
        """

        assert preprocess_strategy == 'z_score' or \
               preprocess_strategy == 'center' or \
               preprocess_strategy == 'raw', "Preprocessing \
        must be either 'center', 'z_score' or 'raw'. Received {0}".format(preprocess_strategy)

        self.preprocess_strategy = preprocess_strategy
        self._fitted = False
        self._n_components = None
        self._U = None
        self._explained_variance = None
        self._mean = None
        self._std = None

    def _preprocess(self, data) -> (torch.Tensor, torch.Tensor):
        if self.preprocess_strategy == 'z_score':
            std, mean = torch.std_mean(data, dim=-2, unbiased=True, keepdim=True)
            return std, mean
        elif self.preprocess_strategy == 'center':
            mean = torch.mean(data, dim=-2, keepdim=True)
            std = torch.ones_like(mean)
            return std, mean
        elif self.preprocess_strategy == 'raw':
            mean = torch.zeros_like(data[0, :])
            std = torch.ones_like(mean)
            return std, mean

    def _get_q(self, n_components: Union[int, float], p: int) -> int:
        if isinstance(n_components, int) and (0 < n_components <= p):
            return n_components
        elif isinstance(n_components, float) and (0.0 < n_components <= 1.0):
            indicator = self._explained_variance > n_components
            values, counts = torch.unique_consecutive(indicator, return_counts=True)
            return counts[0]
        else:
            raise Exception("n_components needs to be an integer in (0, {0}] or a float in (0.0, 1.0). \
            Received {1}".format(p, n_components))

    def fit(self, data) -> "SmartPca":
        if isinstance(data, numpy.ndarray):
            data = torch.tensor(data)
        data = data.float()  # upgrade to full precision in case you are at Half

        std, mean = self._preprocess(data)
        data = (data - mean) / std

        # move stuff to cpu b/c covariance matrix can lead to CUDA out of memory error
        data = data.cpu()
        cov = torch.einsum('np,nq -> pq', data, data) / (data.shape[0] - 1)  # (p x p) covariance matrix
        # add a small diagonal term to make sure that the covariance matrix is not singular
        eps = 1E-4 * torch.randn(cov.shape[0], dtype=cov.dtype, device=cov.device)
        cov += torch.diag_embed(eps, offset=0, dim1=- 2, dim2=- 1)
        try:
            U, S, Vh = torch.linalg.svd(cov, full_matrices=True)
            self._explained_variance = torch.cumsum(S, dim=-1) / torch.sum(S, dim=-1)
            self._U = U
        except RuntimeError as e:
            print("error in torch.svd ->", e)
            self._explained_variance = torch.linspace(start=0.0, end=1.0, steps=cov.shape[0],
                                                      device=cov.device, dtype=cov.dtype)
            self._U = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)

        self._std = std
        self._mean = mean
        self._fitted = True
        return self

    def transform(self, data, n_components: Union[int, float] = None) -> numpy.ndarray:
        """
        Args:
            data: tensor of shape (n, p)
            n_components: If integer specifies the dimensionality of the data after PCA. If float in (0, 1)
                it auto selects the dimensionality so that the explained variance is at least that value.
                If none it uses the previously used value.
        """
        assert self._fitted, "PCA is not fitted. Cal 'fit' or 'fit_transform' first."

        if isinstance(data, numpy.ndarray):
            data = torch.tensor(data)
        data = data.float()  # upgrade to full precision in case you are at Half

        data = (data - self._mean.to(data.device)) / self._std.to(data.device)
        if n_components is None and self._n_components is not None:
            q = self._n_components
            # print("Setting n_components to the previously defined value {0}".format(q))
        elif n_components is None and self._n_components is None:
            raise Exception("n_components has never been specified. Expected n_components = Union[ont, float]")
        else:
            q = self._get_q(n_components, p=data.shape[-1])
            self._n_components = q

        self._U = self._U.to(data.device)
        return torch.einsum('np,pq -> nq', data.float(), self._U[:, :q]).cpu().numpy()

    def fit_transform(self, data, n_components: Union[int, float] = None) -> numpy.ndarray:
        """
        Args:
            data: tensor of shape (n, p)
            n_components: If integer specifies the dimensionality of the data after PCA. If float in (0, 1)
                it auto selects the dimensionality so that the explained variance is at least that value.
                If none (defaults) uses the value previously used.
        """
        if isinstance(data, numpy.ndarray):
            data = torch.tensor(data)
        data = data.float()  # upgrade to full precision in case you are at Half

        self.fit(data)
        return self.transform(data, n_components)


class SmartScaler:
    """ Scale the values using the median and quantiles (with are robust version of mean and variance).
        If clamp=True, each feature is clamped to the quantile range before applying the transformation.
        This is a simple way to deal with the outliers.

        It does not deal with the situation in which outliers are inside the acceptable range
        but very off the reduced manifold as the situation shown below:
                     x  x
                x  x
           x  x        x
        x x
    """

    def __init__(self, quantiles: Tuple[float, float], clamp: bool):
        """
        Args:
            quantiles: Tuple[float, float]. The lowest and largest quantile used to scale the data.
                Must be in (0.0, 1.0)
            clamp: bool. If True, the data is clamped into q_low, q_high before scaling.
        """

        # Inputs
        self._q_low = quantiles[0]
        self._q_high = quantiles[1]
        self._clamp = clamp

        # Things computed during fit
        self._fitted = False
        self._median = None
        self._low = None
        self._high = None

    def _preprocess(self, data) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        median = torch.median(data, dim=-2)[0]
        q_tensor = torch.tensor([self._q_low, self._q_high], dtype=data.dtype, device=data.device)
        qs = torch.quantile(data, q=q_tensor, dim=-2)
        return qs[0], qs[1], median

    def fit(self, data) -> "SmartScaler":
        if isinstance(data, numpy.ndarray):
            data = torch.tensor(data)
        data = data.float()  # upgrade to full precision in case you are at Half

        low, high, median = self._preprocess(data)
        test = low < high
        assert torch.all(test).item()

        self._high = high
        self._low = low
        self._median = median
        self._fitted = True
        return self

    def transform(self, data) -> numpy.ndarray:
        """
        Args:
            data: tensor of shape (n, p)
        """
        assert self._fitted, "Scaler is not fitted. Cal 'fit' or 'fit_transform' first."

        if isinstance(data, numpy.ndarray):
            data = torch.tensor(data)
        data = data.float()  # upgrade to full precision in case you are at Half

        if self._clamp:
            data = data.clamp(min=self._low.to(data.device), max=self._high.to(data.device))

        scale = (self._high - self._low).to(data.device)
        data = (data - self._median.to(data.device)) / scale

        return data.cpu().numpy()

    def fit_transform(self, data) -> numpy.ndarray:
        """
        Args:
            data: tensor of shape (n, p)
        """
        if isinstance(data, numpy.ndarray):
            data = torch.tensor(data)
        data = data.float()  # upgrade to full precision in case you are at Half

        self.fit(data)
        return self.transform(data)




