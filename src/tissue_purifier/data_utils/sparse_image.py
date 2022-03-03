from typing import List, Optional, Tuple, Union
import numpy
import copy
import torch
from tissue_purifier.model_utils.analyzer import SpatialAutocorrelation
from tissue_purifier.data_utils.dataset import CropperSparseTensor
from scanpy import AnnData
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tissue_purifier.data_utils.datamodule import AnndataFolderDM


class SparseImage:
    """
    Sparse torch tensor containing the spatial data
    (for example spatial gene expression or spatial cell types).

    It has 3 dictionaries with spot, patch and image properties.
    """

    def __init__(
        self,
        spot_properties_dict: dict,
        x_key: str,
        y_key: str,
        category_key: str,
        categories_to_codes: dict,
        pixel_size: float,
        padding: int = 10,
        patch_properties_dict: dict = None,
        image_properties_dict: dict = None,
        anndata: AnnData = None,
    ):
        """
        The user can initialize a SparseImage using this constructor or the :method:'from_anndata'.

        Args:
            spot_properties_dict: the dictionary with the spot properties (at the minimum x,y,category)
            x_key: str, the key where the x_coordinates are stored in the spot_properties_dict
            y_key: str, the key where the y_coordinates are stored in the spot_properties_dict
            category_key: str, the key where the category are stored in the spot_properties_dict
            categories_to_codes: dictionary with the mapping from categories (keys) to codes (values).
                The codes must be integers starting from zero. For example {"macrophage" : 0, "t-cell": 1}.
            pixel_size: float, size of the pixel. It used in the conversion
                between raw coordinates and pixel coordinates.
            padding: int, padding of the image (must be >= 1)
            patch_properties_dict: the dictionary with the patch properties.
                If None (defaults) an empty dict is generated.
            image_properties_dict: the dictionary with the image properties.
                If None (defaults) an empty dict is generated.
            anndata: the anndata object with other information (such as count_matrix etc)
        """
        # These variable are passed in
        self._padding = max(1, int(padding))
        self._pixel_size = pixel_size
        self._x_key = x_key
        self._y_key = y_key
        self._cat_key = category_key
        self._categories_to_codes = categories_to_codes
        self._spot_properties_dict = spot_properties_dict
        self._patch_properties_dict = {} if patch_properties_dict is None else patch_properties_dict
        self._image_properties_dict = {} if image_properties_dict is None else image_properties_dict
        self._anndata = anndata

        # These variables are built
        self.origin = None
        self.data = self._create_torch_sparse_image(padding=self._padding, pixel_size=self._pixel_size)

        print("The dense shape of the image is ->", self.data.size())
        print("Occupacy (zero, single, double, ...) of voxels in 3D sparse array ->",
              torch.bincount(self.data.values()).cpu().numpy())
        tmp_sp = torch.sparse.sum(self.data, dim=-3)  # sum over categories
        print("Occupacy (zero, single, double, ...) of voxels  in 2D sparse array (summed over category) ->",
              torch.bincount(tmp_sp.values()).cpu().numpy())

    def _create_torch_sparse_image(self, padding: int, pixel_size: float) -> torch.sparse.Tensor:

        # Check all vectors are 1D and of the same length
        x_raw = torch.from_numpy(self.x_raw).float()
        y_raw = torch.from_numpy(self.y_raw).float()
        cat_raw = self.cat_raw
        assert x_raw.shape == y_raw.shape == cat_raw.shape and len(x_raw.shape) == 1, \
            "Error. x_raw, y_raw, cat_raw must be 1D array of the same length"

        # Check all category are included in categories_to_code
        assert set(cat_raw).issubset(set(self._categories_to_codes.keys())), \
            "Error. Some categories are NOT present in the categories_to_codes dictionary"

        codes = torch.tensor([self._categories_to_codes[cat] for cat in cat_raw]).long()

        # Use GPU if available
        if torch.cuda.is_available():
            x_raw = x_raw.cuda().float()
            y_raw = y_raw.cuda().float()
            codes = codes.cuda().long()

        assert x_raw.shape == y_raw.shape == codes.shape
        assert len(codes.shape) == 1
        print("number of elements --->", codes.shape[0])
        mean_spacing, median_spacing = self.__check_mean_median_spacing__(x_raw, y_raw)
        print("mean and median spacing {0}, {1}".format(mean_spacing, median_spacing))

        # Check that all codes are used at least once
        n_codes = numpy.max(list(self._categories_to_codes.values()))
        code_usage = torch.bincount(codes, minlength=n_codes+1)
        if not torch.prod(code_usage > 0):
            print("WARNING: some codes are not used! \
            This might be OK if some codes correspond to very rare genes or cell_types")

        # Define the raw coordinates of the origin
        x_raw_min = torch.min(x_raw).item()
        y_raw_min = torch.min(y_raw).item()
        self.origin = (x_raw_min - padding * pixel_size, y_raw_min - padding * pixel_size)

        # Convert the coordinates and round to the closest integer
        x_pixel, y_pixel = self.raw_to_pixel(x_raw, y_raw)
        ix = torch.round(x_pixel).long()
        iy = torch.round(y_pixel).long()

        # Create a sparse array with 1 in the correct channel and x,y location.
        # The coalesce make sure that if in case of a collision the values are summed.
        dense_shape = (
            n_codes + 1,
            torch.max(ix).item() + 1 + 2 * padding,
            torch.max(iy).item() + 1 + 2 * padding,
        )
        return torch.sparse_coo_tensor(
            indices=torch.stack((codes, ix, iy), dim=0),
            values=torch.ones_like(codes).int(),
            size=dense_shape,
            device=codes.device,
            requires_grad=False,
        ).coalesce()

    def trim_spot_dictionary(self, keys: List[str]):
        """
        Clear selective entries in the spot_properties_dictionary.

        Args:
            keys: the list of keys to remove from the spot dictionary
        """
        if not isinstance(keys, list):
            keys = [keys]
        for key in keys:
            _ = self.spot_properties_dict.pop(key, None)

    def trim_patch_dictionary(self, keys: List[str]):
        """
        Clear selective entries in the patch_properties_dictionary.

        Args:
            keys: the list of keys to remove from the patch dictionary
        """
        if not isinstance(keys, list):
            keys = [keys]
        for key in keys:
            _ = self.patch_properties_dict.pop(key, None)

    def trim_image_dictionary(self, keys: List[str]):
        """
        Clear selective entries in the image_properties_dictionary.

        Args:
            keys: the list of keys to remove from the image dictionary
        """
        if not isinstance(keys, list):
            keys = [keys]
        for key in keys:
            _ = self.image_properties_dict.pop(key, None)

    def clear(self, patch_dict: bool = True, image_dict: bool = True):
        """
        Clear the patch_properties_dict and image_properties_dict in their entirety.
        Useful to restart the analysis from scratch.
        It will never modify the spot_properties_dict.

        Argsalready present in spot_properties_dict.:
            patch_dict: If True (defaults) the patch_properties_dictionary is cleared
            image_dict: If True (defaults) the image_properties_dictionary is cleared
        """
        if patch_dict:
            self._patch_properties_dict = {}
        if image_dict:
            self._image_properties_dict = {}

    def pixel_to_raw(
            self,
            x_pixel: torch.Tensor,
            y_pixel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the pixel coordinates to the raw_coordinates. This is a simple scale and shift transformation.

        Args:
            x_pixel: tensor of arbitrary shape with the x_index of the pixels
            y_pixel: tensor of arbitrary shape with the x_index of the pixels

        Returns:
            Tuple with the raw coordinates (x_raw and y_raw). They have the same shape as inputs.
        """
        x_raw = x_pixel * self._pixel_size + self.origin[0]
        y_raw = y_pixel * self._pixel_size + self.origin[1]
        return x_raw, y_raw

    def raw_to_pixel(
            self,
            x_raw: torch.Tensor,
            y_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the raw_coordinates to pixel_coordinates. This is a simple scale and shift transformation.

        Args:
            x_raw: tensor of arbitrary shape with the x_raw coordinates
            y_raw: tensor of arbitrary shape with the y_raw coordinates

        Returns:
            Tuple with the pixel coordinates (x_pixel and y_pixel). They have the same shape as inputs.
        """
        x_pixel = (x_raw - self.origin[0]) / self._pixel_size
        y_pixel = (y_raw - self.origin[1]) / self._pixel_size
        return x_pixel, y_pixel

    @property
    def anndata(self):
        """ Return the ann data object """
        return self._anndata

    @property
    def spot_properties_dict(self) -> dict:
        """ Return the spot properties dictionary """
        return self._spot_properties_dict

    @property
    def image_properties_dict(self) -> dict:
        """ Return the image property dictionary """
        return self._image_properties_dict

    @property
    def patch_properties_dict(self) -> dict:
        """ Return the patch property dictionary """
        return self._patch_properties_dict

    @property
    def x_raw(self) -> numpy.ndarray:
        """ Extract the x_coordinates from the original data """
        return numpy.asarray(self.spot_properties_dict[self._x_key])

    @property
    def y_raw(self) -> numpy.ndarray:
        """ Extract the y_coordinates from the original data """
        return numpy.asarray(self.spot_properties_dict[self._y_key])

    @property
    def cat_raw(self) -> numpy.ndarray:
        """ Extract the category for the original data and return them as numpy tensor (as they might contain string)"""
        return numpy.asarray(self.spot_properties_dict[self._cat_key])

    @property
    def n_spots(self) -> int:
        return self.x_raw.shape[0]

    @property
    def shape(self):
        return self.data.shape

    @property
    def device(self):
        return self.data.device

    @property
    def is_sparse(self) -> bool:
        return self.data.is_sparse

    def coalesce(self) -> "SparseImage":
        self.data = self.data.coalesce()
        return self

    def size(self) -> torch.Size:
        return self.data.size()

    def to(self, *args, **kwargs) -> "SparseImage":
        """ Move the data to a device or cast it to a different type """
        self.data = self.data.to(*args, **kwargs)
        return self

    def cuda(self) -> "SparseImage":
        self.data = self.data.cuda()
        return self

    def cpu(self) -> "SparseImage":
        self.data = self.data.cpu()
        return self

    def indices(self) -> torch.Tensor:
        return self.data.indices()

    def values(self) -> torch.Tensor:
        return self.data.values()

    def inspect(self):
        """ Describe the content of the spot, patch and image properties dictionaries """

        def _inspect_dict(d, prefix: str = ''):
            for k, v in d.items():
                if isinstance(v, list):
                    print(prefix, k, type(v), len(v))
                elif isinstance(v, torch.Tensor):
                    print(prefix, k, type(v), v.shape, v.device)
                elif isinstance(v, numpy.ndarray):
                    print(prefix, k, type(v), v.shape)
                elif isinstance(v, dict):
                    print(prefix, k, type(v))
                    _inspect_dict(v, prefix=prefix + "-->")
                else:
                    print(prefix, k, type(v))

        print("")
        print("-- spot_properties_dict --")
        _inspect_dict(self.spot_properties_dict)

        print("")
        print("-- patch_properties_dict --")
        _inspect_dict(self.patch_properties_dict)

        print("")
        print("-- image_properties_dict --")
        _inspect_dict(self.image_properties_dict)

    def to_dense(self) -> torch.Tensor:
        """
        Create a dense torch tensor of shape (channel, width, height)
        where the number of channels is equal to the number of categories of
        the underlying spatial data.

        Note:
            This will convert the sparse array into a dense array and might
            lead to a very large memory footprint.

        Note:
            It is useful for visualization of the data.
        """
        return self.data.to_dense()

    def to_rgb(self,
               spot_size: float = 1.0,
               cmap: matplotlib.colors.ListedColormap = None,
               figsize: Tuple = (8, 8),
               show_colorbar: bool = True,
               contrast: float = 1.0) -> (torch.Tensor, matplotlib.pyplot.Figure):
        """
        Make a 3 channel RGB image. Returns tensor and matplotlig figure.

        Args:
            spot_size: size of sigma of gaussian kernel for rendering the spots
            cmap: the colormap to use
            figsize: the size of the figure
            show_colorbar: If True show the colorbar
            contrast: change to increase/decrease the contrast in the figure. It does not affect the returned tensor.

        Returns:
            A torch.Tensor of size (3, width, height) with the rgb rendering of the image
            and a matplotlib figure.
        """

        def _make_kernel(_sigma: float):
            n = int(1 + 2 * numpy.ceil(4.0 * _sigma))
            dx_over_sigma = torch.linspace(-4.0, 4.0, 2 * n + 1).view(-1, 1)
            dy_over_sigma = dx_over_sigma.clone().permute(1, 0)
            d2_over_sigma2 = (dx_over_sigma.pow(2) + dy_over_sigma.pow(2)).float()
            kernel = torch.exp(-0.5 * d2_over_sigma2)
            return kernel

        def _get_color_tensor(_cmap, _ch):
            if _cmap is None:
                # cm = cc.cm.glasbey_bw_minc_20
                cm = plt.get_cmap('tab20', _ch)
                x = numpy.arange(_ch)
                colors_np = cm(x)
            else:
                cm = plt.get_cmap(_cmap, _ch)
                x = numpy.linspace(0.0, 1.0, _ch)
                colors_np = cm(x)

            color = torch.Tensor(colors_np)[:, :3]
            assert color.shape[0] == _ch
            return color

        dense_img = self.to_dense().unsqueeze(dim=0).float()  # shape: (1, ch, width, height)
        ch = dense_img.shape[-3]
        weight = _make_kernel(spot_size).expand(ch, 1, -1, -1)

        if torch.cuda.is_available():
            dense_img = dense_img.cuda()
            weight = weight.cuda()

        dense_rasterized_img = F.conv2d(
            input=dense_img,
            weight=weight,
            bias=None,
            stride=1,
            padding=(weight.shape[-1] - 1) // 2,
            dilation=1,
            groups=ch,
        ).squeeze(dim=0)

        colors = _get_color_tensor(cmap, ch).float().to(dense_rasterized_img.device)
        rgb_img = torch.einsum("cwh,cn -> nwh", dense_rasterized_img, colors)
        in_range_min, in_range_max = torch.min(rgb_img), torch.max(rgb_img)
        dist = in_range_max - in_range_min
        scale = 1.0 if dist == 0.0 else 1.0 / dist
        rgb_img.add_(other=in_range_min, alpha=-1.0).mul_(other=scale).clamp_(min=0.0, max=1.0)
        rgb_img = rgb_img.detach().cpu()

        # make the figure
        fig, ax = plt.subplots(figsize=figsize)
        _ = ax.imshow((rgb_img.permute(1, 2, 0)*contrast).clamp(min=0.0, max=1.0))

        if show_colorbar:
            discrete_cmp = matplotlib.colors.ListedColormap(colors.cpu().numpy())
            normalizer = matplotlib.colors.BoundaryNorm(
                boundaries=numpy.linspace(-0.5, ch - 0.5, ch + 1),
                ncolors=ch,
                clip=True)

            scalar_mappable = matplotlib.cm.ScalarMappable(norm=normalizer, cmap=discrete_cmp)
            cbar = fig.colorbar(scalar_mappable, ticks=numpy.arange(ch), ax=ax)
            legend_colorbar = list(self._categories_to_codes.keys())
            cbar.ax.set_yticklabels(legend_colorbar)
        plt.close()

        return rgb_img, fig

    def crops(
            self,
            crop_size: int,
            strategy: str = 'random',
            n_crops: int = 10,
            n_element_min: int = 0,
            stride: int = 200,
            random_order: bool = True) -> Tuple[List[torch.sparse.Tensor], List[int], List[int]]:
        """
        Wrapper around :class:'CropperSparseTensor'.

        Returns:
            Three lists with crops, x_locations, y_locations of each crop
        """
        return CropperSparseTensor(strategy=strategy,
                                   n_crops=n_crops,
                                   n_element_min=n_element_min,
                                   crop_size=crop_size,
                                   stride=stride,
                                   random_order=random_order)(self.data)

    def moran(
            self,
            n_neighbours: Optional[int] = 6,
            radius: Optional[float] = None,
            neigh_correct: bool = False) -> torch.Tensor:
        """
        Wrapper around :class:'compute_spatial_autocorrelation'.

        Returns:
            torch.tensor of shape C with the Moran's I score for each channels (i.e. how one channel is
            mixed with all the others)
        """
        return SpatialAutocorrelation(modality='moran',
                                      n_neighbours=n_neighbours,
                                      radius=radius,
                                      neigh_correct=neigh_correct)(self.data)

    def gready(
            self,
            n_neighbours: Optional[int] = 6,
            radius: Optional[float] = None,
            neigh_correct: bool = False) -> torch.Tensor:
        """
        Wrapper around :class:'compute_spatial_autocorrelation'.

        Returns:
            torch.tensor of shape C with the Gready score for each channels (i.e. how one channel is
            mixed with all the others)
        """
        return SpatialAutocorrelation(modality='gready',
                                      n_neighbours=n_neighbours,
                                      radius=radius,
                                      neigh_correct=neigh_correct)(self.data)

    def compute_ncv(self,
                    feature_name: str = None,
                    k: int = None,
                    r: float = None,
                    overwrite: bool = False):
        """
        Compute the neighborhood composition vectors (ncv) of every spot
        and store the results in the spot_properties_dictionary under the :attr:'feature_name' key.

        Args:
            feature_name: the key under which the results will be stored.
            k: if specified the k nearest neighbours are used to compute the ncv.
            r: if specified the neighbours at a distance less than r (in raw units) are used to compute the ncv.
            overwrite: if the :attr:'feature_name' is already present in the spot_properties_dict,
                this variable controls when to overwrite it.
        """

        assert (k is None and r is not None) or (k is not None and r is None), \
            "Exactly one between r and k must be defined."
        assert k is None or isinstance(k, int) and k > 0, "k is either None or a positive integer"
        assert r is None or r > 0, "r is either None or a positive value"

        if feature_name in self.spot_properties_dict.keys() and not overwrite:
            print("The key {0} is already present in spot_properties_dict.")
            print(" Set overwrite=True to overwrite its value. Nothing will be done.".format(feature_name))
            return
        elif feature_name in self.spot_properties_dict.keys() and overwrite:
            print("The key {0} is already present in spot_properties_dict and it will be overwritten".format(
                feature_name))

        # preparation
        cell_type_codes = torch.tensor([self._categories_to_codes[cat] for cat in self.cat_raw]).long()
        metric_features = numpy.stack((self.x_raw, self.y_raw), axis=-1)
        chs = self.shape[-3]
        cell_types_one_hot = torch.nn.functional.one_hot(cell_type_codes, num_classes=chs).cpu()  # shape (*, ch)

        if k is not None:
            # use a knn neighbours
            from sklearn.neighbors import KDTree
            kdtree = KDTree(metric_features)
            dist, ind = kdtree.query(metric_features, k=k)  # shapes (*, k), (*, k)
            ncv_tmp = cell_types_one_hot[ind]  # shape (*, k, ch)
            ncv = ncv_tmp.sum(dim=-2)  # shape (*, ch)
        else:
            # use radius neighbours
            from sklearn.neighbors import BallTree
            balltree = BallTree(metric_features)
            inds = balltree.query_radius(metric_features,
                                         r=r,
                                         return_distance=False,
                                         count_only=False,
                                         sort_results=False)
            ncv = torch.zeros_like(cell_types_one_hot)  # shape (*, ch)
            # ind is a list of array of different length. There is noway to avoid the for-loop
            for n, ind in enumerate(inds):
                ncv_tmp = cell_types_one_hot[ind].sum(dim=-2)  # shape (ch)
                ncv[n] = ncv_tmp

        ncv = ncv.float() / ncv.sum(dim=-1, keepdim=True).clamp(min=1.0)
        self.spot_properties_dict[feature_name] = ncv.cpu().numpy()

    def _set_patchxywh_for_key(self, key, value):
        self.patch_properties_dict[key+"_patch_xywh"] = value

    def _get_patchxywh_for_key(self, key):
        return self.patch_properties_dict[key+"_patch_xywh"]

    @torch.no_grad()
    def compute_patch_features(
            self,
            feature_name: str,
            datamodule: AnndataFolderDM,
            model: torch.nn.Module,
            apply_transform: bool = True,
            batch_size: int = 64,
            n_patches_max: int = 100,
            overwrite: bool = False,
            return_crops: bool = False) -> Union[torch.Tensor, None]:
        """
        Split the sparse image into (possibly overlapping) patches.
        Each patch is analyzed by the model.
        The features are stored in the patch_properties_dict.

        Args:
            feature_name: the key under which the results will be stored.
            datamodule: the datamodule used for training the model. This guarantees that the cropping strategy and
                the data augmentations are identical to the one used during training.
            model: the trained model will ingest the patch and produce the features.
            apply_transform: if True (defaults) the datamodule.test_trasform will be applied to the crops before
                feeding them into the model.
                If False no transformation is applied and the sparse tensors are fed into the model.
            batch_size: how many crops to process simultaneously (default = 64)
            n_patches_max: maximum number of patches generated to analyze the current picture (default = 100)
            overwrite: if the :attr:'feature_names' are already present in the patch_properties_dict,
                this variable controls when to overwrite them.
            return_crops: if True the model returns a (batched) torch.Tensor of shape (n_patches_max, ch, w, h)
                with all the crops which were fed to the model. Default is False.

        Returns:
            if :attr:'return_crops' is False returns None.
            else return a (batched) torch.Tensor of shape (n_patches_max, ch, w, h)
        """

        if feature_name in self.patch_properties_dict.keys() and not overwrite:
            print("The key {0} is already present in patch_properties_dict.")
            print(" Set overwrite=True to overwrite its value. Nothing will be done.".format(feature_name))
            return
        elif feature_name in self.patch_properties_dict.keys() and overwrite:
            print("The key {0} is already present in patch_properties_dict and it will be overwritten".format(
                feature_name))

        # set the model into eval mode
        was_original_in_training_mode = model.training
        model.eval()

        all_patches, all_features = [], []
        n_patches = 0
        patches_x, patches_y, patches_w, patches_h = [], [], [], []
        while n_patches < n_patches_max:
            n_tmp = min(batch_size, n_patches_max - n_patches)
            crops, x_locs, y_locs = datamodule.cropper_test(self.data, n_crops=n_tmp)
            patches_x += x_locs
            patches_y += y_locs
            patches_w += [crop.shape[-2] for crop in crops]
            patches_h += [crop.shape[-1] for crop in crops]
            n_patches += len(x_locs)

            if apply_transform:
                patches = datamodule.trsfm_test(crops)
            else:
                patches = crops

            if return_crops:
                all_patches.append(patches.detach().cpu())

            features_tmp = model(patches)
            if isinstance(features_tmp, torch.Tensor):
                all_features.append(features_tmp)
            elif isinstance(features_tmp, numpy.ndarray):
                all_features.append(torch.from_numpy(features_tmp))
            elif isinstance(features_tmp, list):
                all_features += features_tmp
            else:
                raise NotImplementedError

        # put back the model in the state it was original
        if was_original_in_training_mode:
            model.train()

        # make a single batched tensor of both patches coordinates
        x_torch = torch.tensor(patches_x, dtype=torch.int).cpu()
        y_torch = torch.tensor(patches_y, dtype=torch.int).cpu()
        w_torch = torch.tensor(patches_w, dtype=torch.int).cpu()
        h_torch = torch.tensor(patches_h, dtype=torch.int).cpu()
        patches_xywh = torch.stack((x_torch, y_torch, w_torch, h_torch), dim=-1).long()

        # make a single batched tensor of the features
        if len(all_features) == n_patches_max:
            features = torch.stack(all_features, dim=0).cpu()
        else:
            features = torch.cat(all_features, dim=0).cpu()

        self.patch_properties_dict[feature_name] = features.cpu().numpy()
        self._set_patchxywh_for_key(key=feature_name, value=patches_xywh.cpu().numpy())

        if return_crops:
            if len(all_patches) == n_patches_max:
                patches = torch.stack(all_patches, dim=0).cpu()
            else:
                patches = torch.cat(all_patches, dim=0).cpu()
            return patches

    def transfer_patch_to_spot(
            self,
            keys_to_transfer: List[str],
            overwrite: bool = False,
            verbose: bool = False,
            strategy_patch_to_image: str = "average",
            strategy_image_to_spot: str = "bilinear"):
        """
        Convenience function which sequentially transfer annotations from patch -> image -> spot
        """
        if verbose:
            print("transferring annotations from patch to image first")

        self.transfer_patch_to_image(
            keys_to_transfer=keys_to_transfer,
            overwrite=overwrite,
            verbose=verbose,
            strategy=strategy_patch_to_image,
        )

        if verbose:
            print("transferring annotations from image to spot last")

        self.transfer_image_to_spot(
            keys_to_transfer=keys_to_transfer,
            overwrite=overwrite,
            verbose=verbose,
            strategy=strategy_image_to_spot,
        )

    def transfer_patch_to_image(
            self,
            keys_to_transfer: List[str],
            overwrite: bool = False,
            verbose: bool = False,
            strategy: str = "average"):
        """
        Collect the properties computed separately for each patch and stored in patch_properties_dict to create
        an image properties which will be stored in image_properties_dict under the same name.

        Args:
            keys_to_transfer: keys of the quantity to transfer from patch_properties_dict to image_properties_dict.
                The patch_quantity can be: a scalar, a vector, a scalar field or a vector field.
                This corresponds to patch_quantity having shapes:
                (N_patches), (N_patches, ch), (N_patches, w, h) or (N_patches, ch, w, h) respectively.
            overwrite: bool, in case of collision between keys this variable controls when to overwrite the values in
                the image_properties_dict.
            strategy: str, either 'average' (default) or 'closest'. If 'average' the value of each pixel in the image
                is obtained by averaging the contribution of all patches which contain that pixel. If 'nearest' each
                pixel takes the value from the patch whose center is closets to the pixel.
            verbose: bool, if true print intermediate messages
        """

        def _to_torch(_x):
            if isinstance(_x, torch.Tensor):
                return _x
            elif isinstance(_x, numpy.ndarray):
                return torch.from_numpy(_x).float().cpu()
            else:
                raise Exception("Expect torch.Tensor or numpy.ndarray. Received {0}.".format(type(_x)))

        # make sure keys_to_transfer is provided as a list
        if not isinstance(keys_to_transfer, list):
            keys_to_transfer = [keys_to_transfer]

        # check keys_to_transfer are present in the source
        assert set(keys_to_transfer).issubset(self.patch_properties_dict.keys()), \
            "Some keys are not present in patch_properties_dict."

        # check name collision at destination
        keys_at_destination = self.image_properties_dict.keys()
        for key in keys_to_transfer:
            if key in keys_at_destination and not overwrite:
                print("The key {0} is already present in image_properties_dict. \
                        Set overwrite=True to overwrite its value. \
                        Nothing will be done.".format(key))
                return
            elif key in keys_at_destination and overwrite:
                print("The key {0} is already present in image_properties_dict and it will be overwritten".format(
                    key))

        # Here is where the actual calculation starts
        for key in keys_to_transfer:
            if verbose:
                print("working on ->", key)

            patch_quantity = _to_torch(self.patch_properties_dict[key]).to(device=self.device, dtype=torch.float)
            patch_xywh = _to_torch(self._get_patchxywh_for_key(key=key)).to(device=self.device, dtype=torch.long)

            assert patch_quantity.shape[0] == patch_xywh.shape[0], \
                "Shape mismatched for {0}. Received {1} and {2}".format(key, patch_quantity.shape, patch_xywh.shape)

            len_shape = len(patch_quantity.shape)
            if len_shape == 1:
                # scalar property -> (N)
                ch = 1
                patch_quantity = patch_quantity[:, None, None, None]
            elif len_shape == 2:
                # vector property -> (N, ch)
                ch = patch_quantity.shape[-1]
                patch_quantity = patch_quantity[..., None, None]
            elif len_shape == 3:
                # scalar field -> (N,w,h)
                ch = 1
                patch_quantity = patch_quantity.unsqueeze(1)
            elif len_shape == 4:
                # vector field -> (N, ch, w, h)
                ch = patch_quantity.shape[-3]
            else:
                raise Exception("Can not interpret the dimension of the path_property {0} of \
                shape {1}".format(key, patch_quantity.shape))

            w_all, h_all = self.shape[-2:]
            tmp_result = torch.zeros((ch, w_all, h_all), device=patch_quantity.device, dtype=patch_quantity.dtype)
            tmp_counter = torch.zeros((w_all, h_all), device=patch_quantity.device, dtype=torch.int)
            tmp_distance: torch.Tensor = torch.ones((w_all, h_all),
                                                    device=patch_quantity.device,
                                                    dtype=torch.float) * numpy.inf
            for n, xywh in enumerate(patch_xywh):
                x, y, w, h = xywh.unbind(dim=0)
                if strategy == 'average':
                    tmp_counter[x:x+w, y:y+h] += 1
                    tmp_result[:, x:x+w, y:y+h] += patch_quantity[n]
                elif strategy == 'closest':
                    dw_from_center = torch.linspace(start=-0.5 * (w - 1), end=0.5 * (w - 1), steps=w)
                    dh_from_center = torch.linspace(start=-0.5 * (h - 1), end=0.5 * (h - 1), steps=h)
                    d2_from_center: torch.Tensor = dw_from_center[:, None].pow(2) + dh_from_center[None, :].pow(2)
                    mask = (d2_from_center < tmp_distance[x:x + w, y:y + h])  # shape (w, h)

                    # If the current patch has a smaller distance. Overwrite patch_quantity and tmp_distance
                    tmp_result[:, x:x+w, y:y+h] = \
                        torch.where(mask[None], patch_quantity[n], tmp_result[:, x:x+w, y:y+h])
                    tmp_distance[x:x+w, y:y+h] = torch.min(d2_from_center, tmp_distance[x:x+w, y:y+h])
                else:
                    raise ValueError("strategy can only be 'average' or 'closest'. Received {0}".format(strategy))

            if strategy == 'average':
                result = tmp_result / tmp_counter.clamp(min=1.0)
            elif strategy == 'closest':
                result = tmp_result
            else:
                raise ValueError("strategy can only be 'average' or 'closest'. Received {0}".format(strategy))
            self.image_properties_dict[key] = result.cpu().numpy()

    def transfer_image_to_spot(
            self,
            keys_to_transfer: List[str],
            overwrite: bool = False,
            verbose: bool = False,
            strategy: str = "bilinear"):
        """
        Evaluate the image_properties_dict at the spots location.
        Store the results in the spot_properties_dict with the same name.

        Args:
            keys_to_transfer: the keys of the quantity to transfer from image_properties_dict to spot_properties_dict.
            overwrite: bool, in case of collision between the keys this variable controls
                when the value will be overwritten.
            verbose: bool, if true intermediate messages are displayed.
            strategy: str, either 'closest' (default) or 'bilinear'.
        """

        def _to_torch(_x):
            if isinstance(_x, torch.Tensor):
                return _x
            elif isinstance(_x, numpy.ndarray):
                return torch.from_numpy(_x).float().cpu()
            else:
                raise Exception("Expect torch.Tensor or numpy.ndarray. Received {0}.".format(type(_x)))

        def _interpolation(data_to_interpolate, x_float, y_float, _strategy):
            if _strategy == 'closest':
                ix_long, iy_long = torch.round(x_float).long(), torch.round(y_float).long()
                return data_to_interpolate[..., ix_long, iy_long]

            elif _strategy == 'bilinear':

                x1 = torch.floor(x_float).long()
                x2 = torch.ceil(x_float).long()
                y1 = torch.floor(y_float).long()
                y2 = torch.ceil(y_float).long()

                f11 = data_to_interpolate[..., x1, y1]
                f12 = data_to_interpolate[..., x1, y2]
                f21 = data_to_interpolate[..., x2, y1]
                f22 = data_to_interpolate[..., x2, y2]

                w11 = (x2 - x_float) * (y2 - y_float)
                w12 = (x2 - x_float) * (y_float - y1)
                w21 = (x_float - x1) * (y2 - y_float)
                w22 = (x_float - x1) * (y_float - y1)

                den = ((x2 - x1) * (y2 - y1)).float()

                return (w11 * f11 + w12 * f12 + w21 * f21 + w22 * f22) / den

        assert strategy == 'bilinear' or strategy == 'closest', "Invalid interpolation_method \
        Expected 'bilinear' or 'closest'. Received {0}. ".format(strategy)

        # make sure keys_to_transfer is provided as a list
        if not isinstance(keys_to_transfer, list):
            keys_to_transfer = [keys_to_transfer]

        assert set(keys_to_transfer).issubset(set(self.image_properties_dict.keys())), \
            "Some keys are not present in self.image_properties_dict"

        keys_at_destination = self.spot_properties_dict.keys()
        for key in keys_to_transfer:
            if key in keys_at_destination and not overwrite:
                print("The key {0} is already present in spot_properties_dict. \
                        Set overwrite=True to overwrite its value. \
                        Nothing will be done.".format(key))
                return
            elif key in keys_at_destination and overwrite:
                print("The key {0} is already present in spot_properties_dict and it will be overwritten".format(key))

        # actual calculation
        for key in keys_to_transfer:
            if verbose:
                print("working on ->", key)
            image_quantity = _to_torch(self.image_properties_dict[key])

            assert isinstance(image_quantity, torch.Tensor)
            assert len(image_quantity.shape) == 3 and image_quantity.shape[-2:] == self.shape[-2:]

            x_raw = torch.from_numpy(self.x_raw).float()
            y_raw = torch.from_numpy(self.y_raw).float()
            x_pixel, y_pixel = self.raw_to_pixel(x_raw=x_raw, y_raw=y_raw)
            interpolated_values = _interpolation(
                image_quantity, x_pixel, y_pixel, strategy).cpu()

            assert len(interpolated_values.shape) == 2
            self.spot_properties_dict[key] = interpolated_values.permute(dims=(1, 0)).cpu().numpy()

    def get_state_dict(self, include_anndata: bool = True):
        """ Return the dictionary with the state of the system """
        state_dict = {
            'padding': self._padding,
            'pixel_size': self._pixel_size,
            'x_key': self._x_key,
            'y_key': self._y_key,
            'category_key': self._cat_key,
            'categories_to_codes': self._categories_to_codes,
            'spot_properties_dict': self.spot_properties_dict,
            'patch_properties_dict': self.patch_properties_dict,
            'image_properties_dict': self.image_properties_dict,
            'anndata': self._anndata if include_anndata else None,
        }
        return state_dict

    @classmethod
    def from_state_dict(cls, state_dict: dict) -> "SparseImage":
        """
        Create a sparse image from the state_dictionary which was obtained by the :method:'get_state_dict'

        Examples:
            >>> state_dict_v1 = sparse_image_old.get_state_dict(include_anndata=True)
            >>> torch.save(state_dict_v1, "ckpt.pt")
            >>> state_dict_v2 = torch.load("ckpt.pt")
            >>> sparse_image_new = SparseImage.from_state_dict(state_dict_v2)
        """
        sparse_img_obj = cls(**state_dict)
        return sparse_img_obj

    @classmethod
    def from_anndata(
            cls,
            anndata: AnnData,
            x_key: str,
            y_key: str,
            category_key: str,
            pixel_size: float = None,
            categories_to_channels: dict = None,
            padding: int = 10,
    ):
        """
        Create a SparseImage object from an AnnData object.

        Note:
            The minimal adata object must have a categorical variable (such as cell_type or gene_identities)
            and the spatial coordinates. Additional fields can be present.

        Args:
            anndata: the AnnData object with the spatial data
            x_key: str, tha key associated with the x_coordinate in the AnnData object
            y_key: str, tha key associated with the y_coordinate in the AnnData object
            category_key: str, tha key associated with the categorical values (cell_types or gene_identities)
            pixel_size: float, pixel_size used to convert from raw coordinates to pixel coordinates.
                If it is not specified it will be chosen to be 1/3 of the median of the Nearest Neighbour distances
                between spots. Explicitely setting this attribute ensures that the pixel_size will be consistent
                across multiple images
            categories_to_channels: dictionary with the mapping from the names (of cell_types or genes) to channels.
                Explicitely setting this attribute ensures that the encoding between category
                and channels codes will be consistent across multiple images.
                If not given, the mapping will be inferred from the anndata object.
            padding: int, padding of the image so that the image has a bit of black around it

        Returns:
            The sparse image object

        Examples:
            >>> # create an AnnData object and a sparse image from it
            >>> anndata = AnnDatae(obs={"cell_type": cell_type}, obsm={"spatial": spot_xy_coordinates})
            >>> cell_types = ("ES", "Endothelial", "Leydig", "Macrophage", "Myoid", "RS", "SPC", "SPG", "Sertoli")
            >>> categories_to_codes = dict(zip(cell_types, range(len(cell_types))))
            >>> sparse_image = SparseImage.from_anndata(
            >>>     anndata=anndata,
            >>>     x_key="spatial",
            >>>     y_key="spatial",
            >>>     category_key="cell_type",
            >>>     categories_to_channels=categories_to_channels)
            >>>
            >>> anndata = AnnDatae(obs={"gene": gene, "x": gene_location_x, "y": gene_location_y})
            >>> sparse_image = SparseImage.from_anndata(
            >>>     anndata=anndata,
            >>>     x_key="gene_location_x",
            >>>     y_key="gene_location_y",
            >>>     category_key="gene",
            >>>     pixel_size=6.5,
            >>>     padding=8)
        """

        # extract the xy coordinates an category and build a minimal spot_dictionary
        if x_key == y_key:
            coordinates = numpy.asarray(anndata.obsm[x_key])
            assert len(coordinates.shape) == 2 and coordinates.shape[-1] == 2
            x_raw = coordinates[:, 0]
            y_raw = coordinates[:, 1]
        else:
            try:
                x_raw = numpy.asarray(anndata.obs[x_key])
            except Exception as e:
                print(e)
                x_raw = numpy.asarray(anndata.obsm[x_key])

            try:
                y_raw = numpy.asarray(anndata.obs[y_key])
            except Exception as e:
                print(e)
                y_raw = numpy.asarray(anndata.obsm[y_key])

        cat_raw = numpy.asarray(anndata.obs[category_key])
        assert isinstance(x_raw, numpy.ndarray)
        assert isinstance(y_raw, numpy.ndarray)
        assert isinstance(cat_raw, numpy.ndarray)
        assert x_raw.shape == y_raw.shape == cat_raw.shape and len(x_raw.shape) == 1

        spot_dictionary = {
            "x_key": x_raw,
            "y_key": y_raw,
            "cat_key": cat_raw}

        for k in anndata.obs.keys():
            if k not in {x_key, y_key, category_key}:
                spot_dictionary[k] = anndata.obs[k]
        for k in anndata.obsm.keys():
            if k not in {x_key, y_key, category_key}:
                spot_dictionary[k] = anndata.obsm[k]
        # I have transferred all the observation I coukld on the spot_dict

        # check if anndata.uns["sparse_image_state_dict"] is present
        try:
            state_dict = anndata.uns.pop("sparse_image_state_dict")

            sparse_img_object = cls(
                spot_properties_dict=spot_dictionary,
                x_key="x_key",
                y_key="y_key",
                category_key="cat_key",
                categories_to_codes=state_dict["categories_to_codes"],
                pixel_size=state_dict["pixel_size"],
                padding=state_dict["padding"],
                patch_properties_dict=state_dict["patch_properties_dict"],
                image_properties_dict=state_dict["image_properties_dict"],
                anndata=anndata)
            return sparse_img_object

        except KeyError:

            if categories_to_channels is None:
                category_values = list(numpy.unique(cat_raw))
                categories_to_channels = dict(zip(category_values, range(len(category_values))))

            assert set(numpy.unique(cat_raw)).issubset(set(categories_to_channels.keys())), \
                " Error. The adata object contains values which are not present in category_values."

            # Get the pixel_size
            if pixel_size is None:
                mean_dnn, median_dnn = cls.__check_mean_median_spacing__(torch.tensor(x_raw), torch.tensor(y_raw))
                pixel_size = 0.25 * median_dnn

            sparse_img_obj = cls(
                spot_properties_dict=spot_dictionary,
                x_key="x_key",
                y_key="y_key",
                category_key="cat_key",
                categories_to_codes=categories_to_channels,
                pixel_size=pixel_size,
                padding=padding,
                patch_properties_dict=None,
                image_properties_dict=None,
                anndata=anndata,
            )

        return sparse_img_obj

    def to_anndata(self, export_full_state: bool = False, overwrite: bool = False, verbose: bool = False):
        """
        Export the spot_properties (and optionally the entire state dict) to the anndata object.

        Args:
            export_full_state: if True (default is False) the entire state_dict is exported into the anndata.uns
            overwrite: if True (default is Flase) some entries in adata.obs and/or adata.obsm might be overwritten.
                Set to False to avoid overwriting.
            verbose: if True (default is False) it print some intermediate statements

        Returns:
             AnnData: object containing the spot_properties_dict (and optionally the full state).

        Note:
            This will make a copy of the anndata object that was used to create the sparse image (if any)

        Examples:
            >>> adata = sparse_image.to_anndata()
            >>> sparse_image_new = SparseImage.from_anndata(adata, x_key="x", y_key="y", category_key="cell_type")
        """
        def _to_numpy(_v):
            if isinstance(_v, numpy.ndarray):
                return _v
            elif isinstance(_v, torch.Tensor):
                return _v.detach().cpu().numpy()
            elif isinstance(_v, list):
                return numpy.array(_v)
            else:
                raise Exception("Error. Expected torch.Tensor, numpy.array or list. Recieved {0}".format(type(_v)))

        if self.anndata is None:
            minimal_spot_dict = {
                self._x_key: self.x_raw,
                self._y_key: self.y_raw,
                self._cat_key: self.cat_raw,
            }
            adata = AnnData(obs=minimal_spot_dict)
        else:
            adata = copy.deepcopy(self.anndata)

        # add the OTHER spot properties to either adata.obs or adata.obsm
        for k, v in self._spot_properties_dict.items():
            if (k not in self._x_key) and (k not in self._y_key) and (k not in self._cat_key):
                v_np = _to_numpy(v)

                # squeeze if possible
                if v_np.shape[-1] == 1:
                    v_np = v_np[:, 0]

                if verbose:
                    print("working on {0} of shape {1}".format(k, v.shape))

                if len(v_np.shape) == 1:
                    if k in adata.obs.keys() and overwrite:
                        print("adata.obs[{0}] will be overwritten. \
                        To change this behavior set overwrite to True".format(k))
                        adata.obs[k] = v_np
                    elif k in adata.obs.keys() and not overwrite:
                        print("adata.obs[{0}] is alaready present. \
                        Nothing will be done. To change this behavior set overwrite to True")
                    else:
                        adata.obs[k] = v_np

                else:
                    if k in adata.obsm.keys() and overwrite:
                        print("adata.obsm[{0}] will be overwritten. \
                        To change this behavior set overwrite to True".format(k))
                        adata.obsm[k] = v_np
                    elif k in adata.obsm.keys() and not overwrite:
                        print("adata.obsm[{0}] is already present. \
                        Nothing will be done. To change this behavior set overwrite to True")
                    else:
                        adata.obsm[k] = v_np

        # Add everything else to adata.uns
        if export_full_state:
            full_state_dict = self.get_state_dict(include_anndata=False)
            full_state_dict.pop("anndata")
            full_state_dict.pop("spot_properties_dict")
            adata.uns["sparse_image_state_dict"] = full_state_dict

        return adata

    @staticmethod
    def __validate_patch_xywh__(patch_xywh) -> tuple:
        if isinstance(patch_xywh, torch.Tensor):
            assert torch.numel(patch_xywh) == 4 and patch_xywh.dtype == torch.long, \
                "patch_xywh must be a torch.tensor of type long with 4 elements corresponding to x,y,w,h."
            x, y, w, h = patch_xywh.flatten().unbind(dim=0)
        elif isinstance(patch_xywh, tuple):
            assert len(patch_xywh) == 4, "patch_xywh must be a tuple with 4 integer elements corresponding to x,y,w,h."
            x, y, w, h = patch_xywh
            assert isinstance(x, int) and isinstance(y, int) and isinstance(w, int) and isinstance(h, int), \
                "Error. the elements in the patch_xywh tuple must be integers"
        else:
            raise Exception("Invalid patch_xywh. \
            Expected Union[torch.Tensor, Tuple[int, int, int, int]]. Received {0}".format(type(patch_xywh)))
        return x, y, w, h

    @staticmethod
    def __check_mean_median_spacing__(x_raw: torch.Tensor, y_raw: torch.Tensor) -> (float, float):
        """ x,y coordinates raw """
        from sklearn.neighbors import KDTree
        locations = torch.stack((x_raw, y_raw), dim=-1).cpu().numpy()
        kdt = KDTree(locations, leaf_size=30, metric='euclidean')
        dist, index = kdt.query(locations, k=2, return_distance=True)
        dist_nn = dist[:, 1]
        return numpy.mean(dist_nn), numpy.median(dist_nn)
