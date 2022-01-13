from typing import List, Tuple, Union
import torch
import torch.nn.functional as F
import torchvision
import math
from functools import lru_cache


class TransformForList(torch.nn.Module):
    """
    Apply transforms to a list of inputs.
    """
    def __init__(self,
                 transform_before_stack: Union[torchvision.transforms.Compose, torch.nn.Module, None],
                 transform_after_stack: Union[torchvision.transforms.Compose, torch.nn.Module, None],
                 ):
        """
        Args:
            transform_before_stack: these transform are applied one element at the time.
                If random transforms each element is subject to a DIFFERENT transformation.
            transform_after_stack: these transform are applied to the entire stack together.
                If random transform each element is subject to the SAME transformation.
        """
        super().__init__()
        self.transform_before_stack = transform_before_stack
        self.transform_after_stack = transform_after_stack

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]):
        if isinstance(x, torch.Tensor):
            x = [x]

        if self.transform_before_stack is not None:
            y = torch.stack([self.transform_before_stack(tensor) for tensor in x], dim=-4)
        else:
            y = torch.stack(x, dim=-4)

        if self.transform_after_stack is not None:
            return self.transform_after_stack(y)
        else:
            return y

    def __repr__(self):
        return "Before stack -> {0}. After stack -> {1}".format(self.transform_before_stack.__repr__(),
                                                                self.transform_after_stack.__repr__())


class RandomGlobalIntensity(torch.nn.Module):
    """ Single multiplicative factor which multiplies all the channels."""
    def __init__(self, f_min: float, f_max: float):
        super().__init__()
        assert 0 < f_min <= f_max
        assert isinstance(f_min, float)
        assert isinstance(f_max, float)
        self.f_min = f_min
        self.f_max = f_max

    def forward(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor)
        f = self.f_min + (self.f_max - self.f_min) * torch.rand(size=[1], device=x.device)
        return x.mul_(f)

    def __repr__(self):
        return self.__class__.__name__ + '(f_min={0}, f_max={1})'.format(self.f_min, self.f_max)


class RandomVFlip(torch.nn.Module):
    """ Vertical flip (up to down) a tensor or a batch of tensor images with probability pflip. """
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor)
        flip = torch.rand(size=[1]) < self.p
        h = x.shape[-2]
        return x[..., torch.arange(h - 1, -1, -1, device=x.device), :] if flip else x

    def __repr__(self):
        return self.__class__.__name__ + '(pflip={0})'.format(self.p)


class RandomHFlip(torch.nn.Module):
    """ Horizontal flip (left to right) a (list of) tensor or a batch of tensor images with probability pflip. """
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor)
        flip = torch.rand(size=[1]) < self.p
        w = x.shape[-1]
        return x[..., torch.arange(w - 1, -1, -1, device=x.device)] if flip else x

    def __repr__(self):
        return self.__class__.__name__ + '(pflip={0})'.format(self.p)


class DropoutSparseTensor(torch.nn.Module):
    """ Perform dropout on a sparse tensor. """

    def __init__(self, p: float, dropout_rate: Union[float, Tuple[float, float]]):
        """
        Args:
            p: the probability of applying dropout.
            dropout_rate: dropout rate in (0.0, 1.0).
                If Tuple[float,float] a uniform random variable is drawn before applying dropout
        """
        super().__init__()
        self.p = p
        if len(dropout_rate) > 2:
            raise Exception("Expected either a tuple or a float. Reveice {0}".format(type(dropout_rate)))
        elif len(dropout_rate) == 2:
            self.dropout_min = float(dropout_rate[0])
            self.dropout_max = float(dropout_rate[1])
        else:
            self.dropout_min = float(dropout_rate)
            self.dropout_max = float(dropout_rate)

        assert self.dropout_min >= 0
        assert 0 < self.dropout_max < 1
        self.dropout_range = self.dropout_max - self.dropout_min

    def __repr__(self):
        return self.__class__.__name__ + '(p={0}, dropout_rate=({1},{2}))'.format(
            self.p, self.dropout_min, self.dropout_max)

    def forward(self, sp_tensor: torch.sparse.Tensor):
        assert isinstance(sp_tensor, torch.sparse.Tensor)

        tmp = torch.rand(size=[2], device=sp_tensor.device)
        is_active = (tmp[0].item() < self.p)
        dropout_rate = self.dropout_min + self.dropout_range * tmp[1].item()

        if not is_active:
            return sp_tensor
        else:
            values = sp_tensor.values()
            values_new = torch.distributions.binomial.Binomial(total_count=values.float(),
                                                               probs=1.0-dropout_rate,  # prob of surviving dropout
                                                               validate_args=False).sample().int()
            mask_filter = (values_new > 0)

            return torch.sparse_coo_tensor(
                indices=sp_tensor.indices()[:, mask_filter],
                values=values_new[mask_filter],
                size=sp_tensor.size(),
                device=sp_tensor.device,
                requires_grad=False).coalesce()


class SparseToDense(torch.nn.Module):
    """ Transform a sparse tensor into a dense one """
    def __repr__(self):
        return self.__class__.__name__

    def forward(self, sp_tensor: torch.sparse.Tensor):
        assert isinstance(sp_tensor, torch.sparse.Tensor)
        return sp_tensor.to_dense().float()


class Rasterize(torch.nn.Module):
    """ Apply a gaussian blur to all channels of a dense tensor. """

    def __init__(self, sigma: Union[float, List[float], Tuple[float, float]], normalize: bool):
        """
        Args:
            sigma: the sigma of the Gaussian kernel used for rasterization in unit of pixel_size.
                If Tuple[float,float] a uniform random variable is drawn before applying the Gaussian Blur.
        """
        super().__init__()
        if isinstance(sigma, List):
            self.sigma_min = float(sigma[0])
            self.sigma_max = float(sigma[1])
        elif isinstance(sigma, Tuple):
            self.sigma_min = float(sigma[0])
            self.sigma_max = float(sigma[1])
        elif isinstance(sigma, float):
            self.sigma_min = sigma
            self.sigma_max = sigma
        else:
            raise Exception("Invalid sigma. Expected type float or tuple received type {0}".format(type(sigma)))

        assert self.sigma_max >= self.sigma_min
        self.normalize = normalize

    @lru_cache(maxsize=15)
    def kernel(self, sigma: float):
        n = int(1 + 2 * math.ceil(4.0 * sigma))
        dx_over_sigma = torch.linspace(-4.0, 4.0, 2 * n + 1).view(-1, 1)
        dy_over_sigma = dx_over_sigma.clone().permute(1, 0)
        d2_over_sigma2 = (dx_over_sigma.pow(2) + dy_over_sigma.pow(2)).float()
        kernel = torch.exp(-0.5 * d2_over_sigma2)
        if self.normalize:
            return kernel / kernel.sum()
        else:
            return kernel

    def __repr__(self):
        return self.__class__.__name__ + '(sigma=({0},{1}), normalize={2})'.format(self.sigma_min,
                                                                                   self.sigma_max,
                                                                                   self.normalize)

    def _random_sigma(self) -> float:
        return self.sigma_min + (self.sigma_max - self.sigma_min) * torch.rand(size=[1]).item()

    def forward(self, tensor: torch.Tensor):
        assert isinstance(tensor, torch.Tensor)

        channels = tensor.shape[-3]
        weight = self.kernel(sigma=self._random_sigma()).to(tensor.device).expand(channels, 1, -1, -1)

        if len(tensor.shape) == 3:
            rasterized_tensor = F.conv2d(
                input=tensor.float().unsqueeze(dim=0),
                weight=weight,
                bias=None,
                stride=1,
                padding=(weight.shape[-1] - 1) // 2,
                dilation=1,
                groups=channels,
            ).squeeze(dim=0)
        elif len(tensor.shape) == 4:
            rasterized_tensor = F.conv2d(
                input=tensor.float(),
                weight=weight,
                bias=None,
                stride=1,
                padding=(weight.shape[-1] - 1) // 2,
                dilation=1,
                groups=channels,
            )
        else:
            raise Exception(
                "Expected a data of either 3 or 4 dimensions. "
                "Instead I received a data of shape",
                tensor.shape,
            )

        return rasterized_tensor


class RandomStraightCut(torch.nn.Module):
    """
    Draw a random straight line and set all the values on one side of the line to zero thus occluding part of a
    dense tensor.

    Note:
        The current implementation will give imprecise results if the input tensor is not a square.
    """

    PI = 3.141592653589793
    N_SAMPLE = 40

    def __init__(self,
                 p: float = 0.5,
                 occlusion_fraction: Tuple[float, float] = (0.25, 0.45)):
        """
        Args:
            p: Probability of the transform being applied
            occlusion_fraction: The range of allowed occlusions. Need to be in (0,1).
        """
        super().__init__()
        self.p = p
        self.occlusion_min, self.occlusion_max = occlusion_fraction
        assert 0.0 <= self.occlusion_min < self.occlusion_max <= 1.0

    def __repr__(self):
        return self.__class__.__name__ + '(p={0}, occlusion_fraction=({1},{2}))'.format(self.p,
                                                                                        self.occlusion_min,
                                                                                        self.occlusion_max)

    @staticmethod
    def _from_b_to_occlusion_fraction(b_list_sorted, b):
        # Helper function which computes the relation between "b" and occlusion_fraction
        a_first_triangle = 0.5 * (b_list_sorted[1] - b_list_sorted[0])
        a_last_triangle = 0.5 * (b_list_sorted[-1] - b_list_sorted[-2])
        a_trapezoid = 1.0 - a_first_triangle - a_last_triangle

        m0 = b < b_list_sorted[0]
        r0 = torch.zeros_like(b)

        m1 = (b >= b_list_sorted[0]) * (b < b_list_sorted[1])
        r1 = a_first_triangle * ((b - b_list_sorted[0]) / (b_list_sorted[1] - b_list_sorted[0])) ** 2

        m2 = (b >= b_list_sorted[1]) * (b < b_list_sorted[2])
        r2 = a_first_triangle + a_trapezoid * (b - b_list_sorted[1]) / (b_list_sorted[2] - b_list_sorted[1])

        m3 = (b >= b_list_sorted[2]) * (b < b_list_sorted[3])
        r3 = 1.0 - a_last_triangle * ((b_list_sorted[3] - b) / (b_list_sorted[3] - b_list_sorted[2])) ** 2

        m4 = b >= b_list_sorted[3]
        r4 = torch.ones_like(b)

        return m0 * r0 + m1 * r1 + m2 * r2 + m3 * r3 + m4 * r4

    @staticmethod
    def _occlude_tensor(x: torch.Tensor, b_value: float, m: float, corner_id: int):
        assert corner_id == 0 or corner_id == 1 or corner_id == 2 or corner_id == 3

        w, h = x.shape[-2:]
        ix_grid = torch.linspace(0.0, 1.0, w).view(-1, 1).to(x.device)
        iy_grid = torch.linspace(0.0, 1.0, h).view(1, -1).to(x.device)

        if corner_id == 1:
            # reverse ix
            ix_grid = torch.ones_like(ix_grid) - ix_grid
        elif corner_id == 2:
            # reverse iy
            iy_grid = torch.ones_like(iy_grid) - iy_grid
        elif corner_id == 3:
            # reverse both ix and iy
            ix_grid = torch.ones_like(ix_grid) - ix_grid
            iy_grid = torch.ones_like(iy_grid) - iy_grid
        occlusion_mask = (iy_grid > b_value + m * ix_grid)

        return occlusion_mask * x

    def _from_m_to_bvalue(self, m: float):

        # Compute the values of b which makes the line pass trough the corners of a square: (0,0), (0,1), (1,0), (1,1)
        # The formula is simply: b = y - m*x
        b_list = [0.0, 1.0, -m, 1.0 - m]
        b_list.sort()

        try:
            # draw many b values between b_min and b_max
            # and select one that gives me an occlusion in the desired range
            b = torch.linspace(b_list[0], b_list[-1], self.N_SAMPLE, dtype=torch.float)
            f = self._from_b_to_occlusion_fraction(b_list, b)
            mask_select = (f >= self.occlusion_min) * (f <= self.occlusion_max)
            index = torch.randint(low=0, high=mask_select.sum().item(), size=[1], dtype=torch.long)
            b_value = b[mask_select][index].item()
        except RuntimeError:
            # get the minimum b_value, i.e. no occlusion
            b_value = b_list[0]
        return b_value

    def _random_m(self) -> float:
        theta = torch.rand(size=[1]) * 0.25 * self.PI  # random angle in (0,45) degrees
        m = torch.tan(theta)  # this is the coefficient of the line: y = b + m * x
        return m.item()

    def forward(self, tensor: torch.Tensor):
        assert isinstance(tensor, torch.Tensor)
        if torch.rand(size=[1]).item() < self.p:

            theta = torch.rand(size=[1]) * 0.25 * self.PI  # random angle in (0,45) degrees
            m = torch.tan(theta).item()  # this is the coefficient of the line: y = b + m * x
            b = self._from_m_to_bvalue(m=m)  # this is the intercept of the line: y = b + m * x
            corner_id = torch.randint(low=0, high=4, size=[1]).item()  # id indicating which corner to cut off

            return self._occlude_tensor(tensor, b_value=b, m=m, corner_id=corner_id)
        else:
            return tensor


# class Elastic2D(torch.nn.Module):
#     """ Apply a random elastic transformation to a tensor or batch of tensors. """
#
#     def __init__(self,
#                  translate_max: float = 0.05,
#                  length_scale: float = 32.0,
#                  ):
#         """
#         Args:
#             translate_max: float in (0.0, 1.0) the maximum displacement in unit of the image size
#             length_scale: float the length scale of the square exponential kernel
#         """
#         super().__init__()
#         self.length_scale = float(length_scale)
#         self.translate_max = float(translate_max)
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(length_scale={0}, translate_max={1})'.format(self.length_scale,
#                                                                                         self.translate_max)
#
#     @lru_cache(maxsize=15)
#     def _tril(self, w, h, l2, device):
#         """ Make a RBF kernel of size (w x h) with length_squared = l2"""
#         ix_grid = torch.arange(w, device=device).view(-1, 1).expand(w, h)
#         iy_grid = torch.arange(h, device=device).view(1, -1).expand(w, h)
#
#         map_points = torch.stack((ix_grid, iy_grid), dim=-1)  # shape: w, h, 2
#         locations = map_points.flatten(start_dim=0, end_dim=-2)  # shape: w * h, 2
#         d = (locations.unsqueeze(-2) - locations.unsqueeze(-3)).abs()  # (w * h, w * h, 2)
#         d2 = d.pow(2).sum(dim=-1).float()
#         rbf_kernel = (-0.5 * d2 / l2).exp() + 1E-3 * torch.eye(d2.shape[0], device=device)
#         tril = torch.linalg.cholesky(rbf_kernel)
#         return tril
#
#     @staticmethod
#     def _batch_mv(bmat, bvec):
#         """ Performs a batched matrix-vector product, with compatible but different batch shapes. """
#         return torch.matmul(bmat, bvec.unsqueeze(-1)).squeeze(-1)
#
#     def forward(self, x: torch.Tensor, debug: bool = False):
#         image = x.unsqueeze(dim=-4) if len(x.shape) == 3 else x  # now works for both 3D and 4D tensor
#         batch_size, channel, height, width = image.shape
#
#         # Make a grid of size (width, heigth, 2) with value in (-1.0, 1.0)
#         ix_grid = torch.linspace(-1.0, 1.0, width, device=image.device).view(-1, 1).expand(width, height)
#         iy_grid = torch.linspace(-1.0, 1.0, height, device=image.device).view(1, -1).expand(width, height)
#         grid = torch.stack((iy_grid, ix_grid), dim=-3)
#
#         # compute the tridiagonal matrix (it is a stored property)
#         size_max = max(width, height)
#         downsample_tmp = 2 ** math.ceil(math.log(size_max / 100, 2))  # integer and power of 2
#         w_small = width // downsample_tmp
#         h_small = height // downsample_tmp
#         downsample_factor = float(w_small) / width
#         l_downsampled = downsample_factor * self.length_scale
#         l2_downsampled = l_downsampled * l_downsampled
#         tril = self._tril(w_small,
#                           h_small,
#                           l2_downsampled,
#                           device=image.device)  # shape: (N,N) with N = w_small x h_small
#
#         # sample the displacement field from a GP process
#         eps_x = torch.randn((batch_size, w_small * h_small), device=image.device)
#         eps_y = torch.randn((batch_size, w_small * h_small), device=image.device)
#         disp_x = self._batch_mv(tril, eps_x)
#         disp_y = self._batch_mv(tril, eps_y)
#
#         # Remove the mean displacement and normalize by the maximum value:
#         disp_x -= torch.mean(disp_x, dim=-1, keepdim=True)
#         disp_y -= torch.mean(disp_y, dim=-1, keepdim=True)
#         disp_x /= torch.max(disp_x.abs(), dim=-1, keepdim=True)[0]
#         disp_y /= torch.max(disp_y.abs(), dim=-1, keepdim=True)[0]
#
#         # Rescale displacement to desired scale
#         disp_x = (disp_x * self.translate_max).view(batch_size, 1, w_small, h_small)
#         disp_y = (disp_y * self.translate_max).view(batch_size, 1, w_small, h_small)
#
#         # Stack the x,y dispalcements
#         disp_downsampled = torch.cat([disp_y, disp_x], dim=-3)
#         disp = torchvision.transforms.Resize(size=image.shape[-2:],
#                                              interpolation=torchvision.transforms.InterpolationMode.BILINEAR)(disp_downsampled)
#
#         # Finally sample the original image at the displaced coordinates
#         flow_field = (grid + disp).clamp(-1, 1).permute(0, 2, 3, 1)  # shape: batch_size, width, height, 2
#         warped = F.grid_sample(image, flow_field, align_corners=False, mode='bilinear')
#         warped = warped.squeeze(dim=-4) if len(x.shape) == 3 else warped
#
#         if debug:
#             return warped, grid, disp, flow_field
#         else:
#             return warped
#
#
# class ToRgb(torch.nn.Module):
#     """ Convert any image to having 3 channels """
#     def forward(self, x):
#         if x.shape[-3] >= 3:
#             # take the first 3 channels
#             return x[..., :3, :, :]
#         else:
#             # duplicate the first channel 3 times
#             new_shapes = list(x.shape)
#             new_shapes[-3] = 3
#             return x[..., :1, :, :].expand(new_shapes)
#
#     def __repr__(self):
#         return self.__class__.__name__
#
#
# class LargestSquareCrop(torch.nn.Module):
#     """ Get the largest possible square crop fully contained in the image and rescale it to the desired size """
#     def __init__(self, size: int):
#         super().__init__()
#         self.size = size
#
#     def forward(self, x):
#         w, h = x.shape[-2:]
#         crop_size = min(w, h)
#         x1 = torchvision.transforms.functional.center_crop(x, crop_size)
#         return torchvision.transforms.functional.resize(x1, size=self.size, antialias=True)
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(size={0})'.format(self.size)
#