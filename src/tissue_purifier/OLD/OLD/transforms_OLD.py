import numpy
import torch
from typing import Tuple, List, Union
import torch.nn.functional as F
import torchvision


class OldTestTransform(object):
    def __init__(self):
        self.transform = torchvision.transforms.Compose([
            StackTensor(dim=-4),
            RandomGaussianBlur(sigma=(1.0, 5.0)),
            torchvision.transforms.Resize(224),
            ])

    def __call__(self, x):
        return self.transform(x)

    def __repr__(self):
        return self.transform.__repr__()


class OldTrainTransform(object):
    def __init__(self):
        self.transform = torchvision.transforms.Compose([
            DropoutSparseTensor(dropout=(0.0, 0.4)),
            StackTensor(dim=-4),
            RandomGaussianBlur(sigma=(1.0, 5.0)),
            torchvision.transforms.RandomAffine(degrees=180,
                                                scale=(0.75, 1.25),
                                                shear=0.0,
                                                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                                                fill=0),
            RandomIntensity(factor=(0.7, 1.3)),
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.Resize(224),
            ])

    def __call__(self, x):
        return self.transform(x)

    def __repr__(self):
        return self.transform.__repr__()


class StackTensor(object):
    """ Stack a list of either sparse or dense tensors of shape (channel, width, height) along the dimension=dim.
        The output is always a DENSE tensor of shape: (batch, channel, width, height)
    """
    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, data: Union[torch.Tensor, List[torch.Tensor]]):
        if isinstance(data, torch.Tensor):
            return data.to_dense().float().unsqueeze(dim=self.dim) if data.is_sparse \
                else data.unsqueeze(dim=self.dim)
        elif isinstance(data, list):
            return torch.stack([tmp.to_dense().float() for tmp in data], dim=self.dim) if data[0].is_sparse \
                else torch.stack(data, dim=self.dim)
        else:
            raise Exception("Invalid data type. \
            Expected torch.tensor or list(torch.tensor) received {0}".format(type(data)))

    def __repr__(self):
        return 'OLD_' + self.__class__.__name__ + '(dim={0})'.format(self.dim)


class RandomIntensity(object):
    """ Multiplies all channel intensity by the same random factor in a designated range. """
    def __init__(self, factor: Tuple[float, float]):
        self.fmin = factor[0]
        self.fmax = factor[1]

    def __call__(self, data: torch.Tensor):
        assert isinstance(data, torch.Tensor) and len(data.shape) == 4

        factor = self.fmin + torch.rand(size=[1], dtype=data.dtype, device=data.device) * (self.fmax - self.fmin)
        data_min = torch.min(data)
        data_max = torch.max(data)
        data_normalized = factor * (data - data_min) / (data_max - data_min)
        return data_normalized

    def __repr__(self):
        return 'OLD_' + self.__class__.__name__ + '(factor=({0},{1}))'.format(self.fmin, self.fmax)


class RandomGaussianBlur(object):
    """ Apply a gaussian blur on a dense tensor. All batch and channel dimensions are treated identically. """

    def __init__(self, sigma: Union[float, Tuple[float, float]]):
        """
        Args:
            sigma: the sigma of the Gaussian kernel used for rasterization in unit of pixel_size.
                If Tuple[float,float] a uniform random variable is drawn before applying the Gaussian Blur.
        """
        if isinstance(sigma, Tuple):
            self.sigma_min = sigma[0]
            self.sigma_max = sigma[1]
        elif isinstance(sigma, float):
            self.sigma_min = sigma
            self.sigma_max = sigma
        else:
            raise Exception("invalid sigma. Expected type float or tuple received type {0}".format(type(sigma)))
        assert self.sigma_max >= self.sigma_min
        if self.sigma_max == self.sigma_min:
            self.kernel = self.make_kernel(self.sigma_max)
        else:
            self.kernel = None

    @staticmethod
    def make_kernel(sigma):
        n = int(1 + 2 * numpy.ceil(4.0 * sigma))
        dx_over_sigma = torch.linspace(-4.0, 4.0, 2 * n + 1).view(-1, 1)
        dy_over_sigma = dx_over_sigma.clone().permute(1, 0)
        d2_over_sigma2 = (dx_over_sigma.pow(2) + dy_over_sigma.pow(2)).float()
        kernel = torch.exp(-0.5 * d2_over_sigma2)
        return kernel / kernel.sum()

    def __call__(self, data: Union[torch.Tensor, List[torch.Tensor]]):
        """
        Args:
            data: torch tensor (or list of) corresponding to an image of shape (*, channel, width, height)

        Returns:
            torch tensor (or list of)
        """

        mylist = data if isinstance(data, List) else [data]
        channels = mylist[0].shape[-3]
        weight = None if (self.kernel is None) else self.kernel.expand(channels, 1, -1, -1).to(mylist[0].device)

        result = []
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.rand(size=[len(mylist)])
        for n, tensor in enumerate(mylist):
            if self.kernel is None:
                weight = self.make_kernel(sigma[n]).expand(channels, 1, -1, -1).to(tensor.device)

            if len(tensor.shape) == 3:
                tmp = F.conv2d(input=tensor.float().unsqueeze(dim=0),
                               weight=weight,
                               bias=None,
                               stride=1,
                               padding=(weight.shape[-1]-1)//2,
                               dilation=1,
                               groups=channels).squeeze(dim=0)
            elif len(tensor.shape) == 4:
                tmp = F.conv2d(input=tensor.float(),
                               weight=weight,
                               bias=None,
                               stride=1,
                               padding=(weight.shape[-1]-1)//2,
                               dilation=1,
                               groups=channels)
            else:
                raise Exception("Expected a tensor of either 3 or 4 dimensions. \
                                 Instead I received a tensor of shape", tensor.shape)
            result.append(tmp)

        return result if isinstance(data, List) else result[0]

    def __repr__(self):
        return 'OLD_' + self.__class__.__name__ + '(sigma=({0},{1}))'.format(self.sigma_min, self.sigma_max)


class DropoutSparseTensor(object):
    """ Perform dropout on a sparse tensor. """

    def __init__(self, dropout: Union[float, Tuple[float, float]]):
        """
        Args:
            dropout: dropout rate in (0.0, 1.0).
                If Tuple[float,float] a uniform random variable is drawn before applying dropout
        """
        if isinstance(dropout, Tuple) or isinstance(dropout, List):
            self.dropout_min = dropout[0]
            self.dropout_max = dropout[1]
        elif isinstance(dropout, float):
            self.dropout_min = dropout
            self.dropout_max = dropout
        else:
            raise Exception("Invalid dropout. Expected type float or tuple or list \
            received type {0}".format(type(dropout)))

        assert self.dropout_min >= 0
        assert self.dropout_max < 1
        self.dropout_range = self.dropout_max - self.dropout_min

    def __call__(self, data: Union[torch.sparse.Tensor, List[torch.sparse.Tensor]]):
        """
        Args:
            data: torch sparse tensor (or list of) corresponding to an image of shape (channel, width, height)

        Returns:
            torch sparse tensor (or list of)
        """

        mylist = data if isinstance(data, List) else [data]

        dropout = self.dropout_min + self.dropout_range * torch.rand(size=[len(mylist)])

        result = []
        for n, sparse_tensor in enumerate(mylist):
            codes, x_pixel, y_pixel = sparse_tensor.indices()
            values = sparse_tensor.values()
            mask = (torch.rand(size=values.size()) > dropout[n])

            result.append(torch.sparse_coo_tensor(indices=torch.stack((codes[mask],
                                                                       x_pixel[mask],
                                                                       y_pixel[mask]), dim=0),
                                                  values=values[mask],
                                                  size=sparse_tensor.size(),
                                                  device=codes.device,
                                                  requires_grad=False).coalesce())

        return result if isinstance(data, List) else result[0]

    def __repr__(self):
        return 'OLD_' + self.__class__.__name__ + '(dropout=({0},{1}))'.format(self.dropout_min, self.dropout_max)


class RandomCropSparseTensor(object):
    """ Perform crops of a sparse tensor. """

    def __init__(self,
                 crop_size: int,
                 n_crops: int = 1,
                 n_element_min: int = 10,
                 safety_factor: int = 10):
        """
        Args:
            crop_size: int, the size in pixel of the random crop
            n_crops: int, the number of random crop to generate. Default = 1
            n_element_min: int, the minimum values of element in the random crop.
                Random crops with too few elements are disregarded.
            safety_factor: int, it generates extra crops b/c some of them will be filtered out
        """
        assert isinstance(crop_size, int) and isinstance(n_crops, int) and \
               isinstance(n_element_min, int) and isinstance(safety_factor, int)
        self.crop_size = crop_size
        self.n_crops = n_crops
        self.n_element_min = n_element_min
        self.safety_factor = safety_factor

    def __call__(self, data: Union["SparseImage", List["SparseImage"], torch.sparse.Tensor, List[torch.sparse.Tensor]]):
        """
        Create many random crops of a torch sparse tensor (or list of).

        Args:
            data: torch sparse tensor (or list of)

        Returns:
            torch sparse tensor (or list of)
        """
        mylist = data if isinstance(data, List) else [data]

        result = []
        for n, sparse_tensor in enumerate(mylist):

            codes, x_pixel, y_pixel = sparse_tensor.indices()
            values = sparse_tensor.values()

            # Generate possible value for the bottom-left corner of the crop
            x_corner = torch.randint(low=0,
                                     high=sparse_tensor.shape[-2] - self.crop_size,
                                     size=[self.n_crops * self.safety_factor],
                                     device=x_pixel.device,
                                     dtype=x_pixel.dtype).view(-1, 1)  # low is included, high is excluded
            y_corner = torch.randint(low=0,
                                     high=sparse_tensor.shape[-1] - self.crop_size,
                                     size=[self.n_crops * self.safety_factor],
                                     device=y_pixel.device,
                                     dtype=y_pixel.dtype).view(-1, 1)  # low is included, high is excluded

            # check if the random crop has a the minimum number of elements in it
            tmp_mask = (x_pixel >= x_corner) * (x_pixel < x_corner + self.crop_size) * \
                       (y_pixel >= y_corner) * (y_pixel < y_corner + self.crop_size)
            n_elements = (values * tmp_mask).sum(dim=-1)
            valid_crop = (n_elements >= self.n_element_min)
            n_valid_crops = valid_crop.sum()

            assert n_valid_crops >= self.n_crops, \
                "Insufficient number of valid crops. \
                Increase the safety_factor and/or decrease n_element_min and/or increase crop_size."

            ix = x_corner[valid_crop, 0][:self.n_crops]  # shape: n_crops
            iy = y_corner[valid_crop, 0][:self.n_crops]  # shape: n_crops
            mask = tmp_mask[valid_crop][:self.n_crops]  # shape: n_crops, element_in_sparse_array
            
            dense_crop_shape = (sparse_tensor.shape[-3], self.crop_size, self.crop_size)

            result += [torch.sparse_coo_tensor(indices=torch.stack((codes[mask[n]],
                                                                    x_pixel[mask[n]] - ix[n],
                                                                    y_pixel[mask[n]] - iy[n]), dim=0),
                                               values=values[mask[n]],
                                               size=dense_crop_shape,
                                               device=codes.device,
                                               requires_grad=False).coalesce() for n in range(self.n_crops)]

        return result if len(result) > 1 else result[0]

    def __repr__(self):
        return 'OLD_' + self.__class__.__name__ + '(crop_size={0}, \
        n_crops={1}, n_element_min={2}))'.format(self.crop_size, self.n_crops, self.n_element_min)
