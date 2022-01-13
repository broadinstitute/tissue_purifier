from typing import Optional
import torch


class NoisyLoss(torch.nn.Module):
    def __init__(self, gamma: float = 0.5, active_coeff: float = 1.0, passive_coeff: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.active_coeff = active_coeff
        self.passive_coeff = passive_coeff

        self.active_loss = FocalLoss(gamma=gamma, weight=None, reduction="mean", normalized=True)
        self.passive_loss = ReversedCrossEntropyLoss(
            weight=None, reduction="mean", log_zero=-4.0, normalized=False
        )

    def forward(self, inputs, targets):
        """
        Args:
            inputs: float torch tensor of size (*, K) where K is the number of classes with the raw activations
            targets: integer tensor of shape (*) with entries 0,1,...,K-1

        Return:
             A scalar if reduction is mean or sum. A vector if reduction is None
        """
        return self.active_coeff * self.active_loss(
            inputs, targets
        ) + self.passive_coeff * self.passive_loss(inputs, targets)


class CrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        reduction: Optional[str] = "mean",
        normalized: bool = False,
    ):
        super().__init__()
        self.weight = weight
        assert (reduction == "mean") or (reduction is None) or (reduction == "sum")
        self.reduction = reduction
        self.normalized = normalized

    def forward(self, inputs, targets):
        """
        Args:
            inputs: float torch tensor of size (*, K) where K is the number of classes with the raw activations
            targets: integer tensor of shape (*) with entries 0,1,...,K-1

        Return:
             A scalar if reduction is mean or sum. A vector if reduction is None
        """
        inputs_select = torch.index_select(inputs, dim=-1, index=targets)
        numerator = -inputs_select + torch.logsumexp(inputs, dim=-1, keepdim=False)
        if self.weight is not None:
            numerator *= self.weight[targets]

        if self.normalized:
            denominator = -torch.nn.LogSoftmax(dim=-1)(inputs).sum(dim=-1)
            tmp = numerator / denominator
        else:
            tmp = numerator

        if self.reduction == "mean":
            return tmp.mean()
        elif self.reduction == "sum":
            return tmp.sum()
        else:
            return tmp


class ReversedCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        reduction: Optional[str] = "mean",
        log_zero: float = -4.0,
        normalized: bool = False,
    ):
        """ See https://arxiv.org/pdf/2006.13554.pdf """
        super().__init__()
        self.normalized = normalized
        self.log_zero = log_zero
        self.weight = weight
        assert (reduction == "mean") or (reduction is None) or (reduction == "sum")
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: float torch tensor of size (*, K) where K is the number of classes with the raw activations
            targets: integer tensor of shape (*) with entries 0,1,...,K-1

        Return:
             A scalar if reduction is mean or sum. A vector if reduction is None
        """
        p = torch.nn.Softmax(dim=-1)(inputs)  # shape (*, K)
        log_q = self.log_zero * torch.ones_like(p)
        log_q.scatter_(dim=-1, index=targets[:, None], src=torch.zeros_like(inputs))

        if self.weight is None:
            tmp = -torch.sum(p * log_q, dim=-1)
        else:
            tmp = -torch.sum(self.weight * p * log_q, dim=-1)

        if self.normalized:
            tmp /= self.log_zero * (inputs.shape[-1] - 1)

        if self.reduction == "mean":
            return tmp.mean()
        elif self.reduction == "sum":
            return tmp.sum()
        else:
            return tmp


class FocalLoss(torch.nn.Module):
    def __init__(
        self,
        gamma: float = 0.25,
        weight: Optional[torch.Tensor] = None,
        reduction: Optional[str] = "mean",
        normalized: bool = False,
    ):
        super().__init__()
        self.normalized = normalized
        self.gamma = gamma
        self.weight = weight
        assert (reduction == "mean") or (reduction is None) or (reduction == "sum")
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: float torch tensor of size (*, K) where K is the number of classes with the raw activations
            targets: integer tensor of shape (*) with entries 0,1,...,K-1

        Return:
             A scalar if reduction is mean or sum. A vector if reduction is None
        """
        log_p = torch.nn.LogSoftmax(dim=-1)(inputs)  # shape (*, K)
        one_mins_p = torch.ones_like(inputs) - torch.nn.Softmax(dim=-1)(inputs)  # shape (*, K)
        if self.weight is None:
            result = (one_mins_p ** self.gamma) * log_p
        else:
            result = self.weight * (one_mins_p ** self.gamma) * log_p

        numerator = -torch.index_select(result, dim=-1, index=targets)
        denominator = -torch.sum(result, dim=-1)
        tmp = numerator / denominator if self.normalized else numerator

        if self.reduction == "mean":
            return tmp.mean()
        elif self.reduction == "sum":
            return tmp.sum()
        else:
            return tmp
