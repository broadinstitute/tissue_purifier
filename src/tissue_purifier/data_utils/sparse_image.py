from typing import List, Optional, Tuple, Callable, Union
import numpy
import copy
import torch
from tissue_purifier.model_utils.analyzer import SpatialAutocorrelation
from tissue_purifier.data_utils.dataset import CropperSparseTensor
from scanpy import AnnData
import matplotlib.colors
import matplotlib.pyplot as plt
import torch.nn.functional as F
import colorcet as cc


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

    def clear_patch_and_image_properties(self):
        """
        Clear the patch_properties_dict and image_properties_dict.
        It is useful you want to restart the analysis from scratch.
        It will not modify the spot_properties_dict.
        """
        self._patch_properties_dict = {}
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
                cm = cc.cm.glasbey_bw_minc_20_maxl_70
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

    @torch.no_grad()
    def analyze_with_tiling(
            self,
            cropper: CropperSparseTensor,
            patch_analyzers: List[Callable],
            feature_names: List[str],
            transform: Callable = None,
            batch_size: int = 64,
            n_patches_max: int = 100,
            store_crops: bool = False,
            store_crops_key: str = None,
            overwrite: bool = False):
        """
        Split the sparse image into (possibly overlapping) patches.
        Each patch is analyzed separately by the patch analyzer.
        The predictions are stored in the patch_properties_dict.

        Args:
            cropper: specify how to split an image into patches
            transform: specify how to convert the sparse_tensor into images before ingestion by a patch_analyzer.
                If None (defaults) no transformation is applied and the sparse tensor is fed into the patch analyzer.
            patch_analyzers: List of objects with the compute_features method (preferred) or  the __call__ method.
                Will be called on each image_patch to extract the features.
            feature_names: List of names of the features to extract.
                Will be used to save the features in patch_properties_dict.
                Must be of the same lengths as patch_analyzers list.
            batch_size: how many crops to process simultaneously (default = 64)
            n_patches_max: maximum number of patches generated to analyze the current picture (default = 100)
            store_crops: if true the crops (after transform) are stored in the patch_dictionary
                under the key :attr:'store_crops_key'.
            store_crops_key: the key associated with the crops. Only used if :attr:'store_crops' == True.
            overwrite: if the :attr:'feature_names' are already present in the patch_properties_dict,
                this variable controls when to overwrite them.
        """
        patch_analyzers = patch_analyzers if isinstance(patch_analyzers, list) else [patch_analyzers]
        feature_names = feature_names if isinstance(feature_names, list) else [feature_names]
        assert len(feature_names) == len(patch_analyzers), \
            "Error, feature_names of length {0} and patch_analyzers of length {1} must have same length".format(
                len(feature_names), len(patch_analyzers))
        assert isinstance(cropper, CropperSparseTensor)

        destination_keys = self._patch_properties_dict.keys()

        all_keys_to_store = feature_names.copy()
        if store_crops:
            all_keys_to_store.append(store_crops_key)

        for k in all_keys_to_store:
            if k in destination_keys and not overwrite:
                print("The key {0} is already present in patch_properties_dict.")
                print(" Set overwrite=True to overwrite its value. Nothing will be done.".format(k))
                return
            if k in destination_keys and overwrite:
                print("The key {0} is already present in patch_properties_dict. This value will be overwritten".format(k))

        patches_xywh = self._patch_properties_dict.get("patch_xywh", None)
        if patches_xywh is None:
            print("I will generate {0} new patches for this sparse image".format(n_patches_max))
        else:
            patches_xywh = patches_xywh.to(device=self.data.device, dtype=torch.long)
            print("I will reuse the {0} patches previously generated for this sparse image".format(patches_xywh.shape[0]))
            n_patches_max = patches_xywh.shape[0]

        n_patches = 0
        patches_x, patches_y, patches_w, patches_h = [], [], [], []
        all_patches = []
        results = dict()
        while n_patches < n_patches_max:
            if patches_xywh is None:
                n_added = min(batch_size, n_patches_max - n_patches)
                crops, x_locs, y_locs = cropper(self.data, n_crops=n_added, strategy='random')
                patches_x += x_locs
                patches_y += y_locs
                patches_w += [crop.shape[-2] for crop in crops]
                patches_h += [crop.shape[-1] for crop in crops]
            else:
                n_added = min(batch_size, n_patches_max - n_patches)
                crops, x_locs, y_locs = cropper.reapply_crops(self.data, patches_xywh[n_patches:n_patches+n_added])
            n_patches += n_added

            if transform is not None:
                patches = transform(crops)
            else:
                patches = crops

            if store_crops:
                all_patches.append(patches.detach().cpu())

            for f_name, analyzer in zip(feature_names, patch_analyzers):
                # defaults to call the __call__ method
                tmp = analyzer(patches)

                # make sure the output is in the form of a list
                if not isinstance(tmp, list):
                    tmp = [tmp]

                if f_name not in results.keys():
                    results[f_name] = tmp
                else:
                    results[f_name] += tmp

        # stack or cat the results in one large tensor
        for key, list_value in results.items():
            if isinstance(list_value, list) and isinstance(list_value[0], torch.Tensor):
                if len(list_value) == n_patches_max:
                    results[key] = torch.stack(list_value, dim=0).cpu()
                else:
                    results[key] = torch.cat(list_value, dim=0).cpu()
            else:
                print(key, type(list_value), type(list_value[0]))
                raise NotImplementedError

        # add the results in patch_properties_dict
        self._patch_properties_dict.update(results)

        # store the patches if necessary
        if store_crops:
            self._patch_properties_dict[store_crops_key] = torch.cat(all_patches, dim=0).detach().cpu()

        # Write the patches coordinates to the dictionary if they are new
        if patches_xywh is None:
            x_torch = torch.tensor(patches_x, dtype=torch.int).cpu()
            y_torch = torch.tensor(patches_y, dtype=torch.int).cpu()
            w_torch = torch.tensor(patches_w, dtype=torch.int).cpu()
            h_torch = torch.tensor(patches_h, dtype=torch.int).cpu()
            patches_xywh = torch.stack((x_torch, y_torch, w_torch, h_torch), dim=-1).long()
            self._patch_properties_dict["patch_xywh"] = patches_xywh

    def patch_property_to_image_property(
            self,
            keys_to_transfer: List[str],
            overwrite: bool = False,
            verbose: bool = False,
            strategy: str = "average"):
        """
        Collect the properties computed separately for each patch and stored in patch_properties_dict to create
        an image properties which will be stored in image_properties_dict with the same name.

        Args:
            keys_to_transfer: keys of the quantity to transfer from patch_properties_dict to image_properties_dict.
                The patch_quantity can be: a scalar, a vector, a scalar field or a vector field.
                This corresponds to patch_quantity having shapes:
                (N_patches), (N_patches, ch), (N_patches, w, h) or (N_patches, ch, w, h) respectively.
            overwrite: bool, in case of collision between keys this variable controls when to overwrite the values in
                the image_properties_dict.
            strategy: str, either 'average' (default) or 'closest'. If 'average' the value of each pixel in the image
                is obtained by averaging the contribution of all patches which contain that pixel. If 'nearest' each
                pixel takes the value from the patch which center is closets to the pixel.
            verbose: bool, if true print intermediate messages
        """

        def _to_torch(_x):
            if isinstance(_x, torch.Tensor):
                return _x
            elif isinstance(_x, numpy.ndarray):
                return torch.from_numpy(_x).float().cpu()
            else:
                raise Exception("The patch quantity to be transfered must be torch.Tensor or numpy.ndarray. \
                Received {0}.".format(type(_x)))

        if not isinstance(keys_to_transfer, list):
            keys_to_transfer = [keys_to_transfer]

        assert set(keys_to_transfer).issubset(self.patch_properties_dict.keys()), \
            "Some keys are not present in self.patch_properties_dict."

        assert "patch_xywh" in self.patch_properties_dict.keys(), \
            " The spot_properties_dict does not have the patch_xywh keyword."
        patch_xywh = self.patch_properties_dict["patch_xywh"].to(device=self.device, dtype=torch.long)

        destination_keys = self.image_properties_dict.keys()
        for k in keys_to_transfer:
            if k in destination_keys and not overwrite:
                print("The key {0} is already present in image_properties_dict. \
                        Set overwrite=True to overwrite its value. \
                        Nothing will be done.".format(k))
                return
            if k in destination_keys and overwrite:
                print("The key {0} is already present in image_properties_dict. \
                        This value will be overwritten".format(k))

        # Here is where the actual calculation starts
        for k in keys_to_transfer:
            if verbose:
                print("working on ->", k)
            patch_quantity = _to_torch(self.patch_properties_dict[k])

            assert patch_quantity.shape[0] == patch_xywh.shape[0], \
                "patch_quantity {0} and patch_xywh must have the same leading dimension. \
                Received {1} and {2}".format(k, patch_quantity.shape, patch_xywh.shape)

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
                shape {1}".format(k, patch_quantity.shape))

            w_all, h_all = self.shape[-2:]
            tmp_result = torch.zeros((ch, w_all, h_all), device=patch_quantity.device, dtype=patch_quantity.dtype)
            tmp_counter = torch.zeros((w_all, h_all), device=patch_quantity.device, dtype=torch.int)
            tmp_distance = torch.ones((w_all, h_all), device=patch_quantity.device, dtype=torch.float) * numpy.inf
            for n, xywh in enumerate(patch_xywh):
                x, y, w, h = xywh.unbind(dim=0)
                if strategy == 'average':
                    tmp_counter[x:x+w, y:y+h] += 1
                    tmp_result[:, x:x+w, y:y+h] += patch_quantity[n]
                elif strategy == 'closest':
                    dw_from_center = torch.linspace(start=-0.5 * (w - 1), end=0.5 * (w - 1), steps=w)
                    dh_from_center = torch.linspace(start=-0.5 * (h - 1), end=0.5 * (h - 1), steps=h)
                    d2_from_center = dw_from_center[:, None].pow(2) + dh_from_center[None, :].pow(2)
                    mask = (d2_from_center < tmp_distance[x:x + w, y:y + h])  # shape (w, h)

                    # If the current patch has a smaller distance. Overwrite patch_quantity and tmp_distance
                    tmp_result[:, x:x+w, y:y+h] = \
                        torch.where(mask[None], patch_quantity[n], tmp_result[:, x:x+w, y:y+h])
                    tmp_distance[x:x+w, y:y+h] = torch.min(d2_from_center, tmp_distance[x:x+w, y:y+h])
                else:
                    raise ValueError("strategy can only be 'average' or 'closest'. Received {0}".format(strategy))

            if strategy == 'average':
                self.image_properties_dict[k] = tmp_result / tmp_counter.clamp(min=1.0)
            elif strategy == 'closest':
                self.image_properties_dict[k] = tmp_result
            else:
                raise ValueError("strategy can only be 'average' or 'closest'. Received {0}".format(strategy))

    def image_property_to_spot_property(
            self,
            keys: List[str],
            overwrite: bool = False,
            verbose: bool = False,
            strategy: str = "closest"):
        """
        Evaluate the image_properties_dict[keys] at the spots location (either cell or genes).
        Store the results in the spot_properties_dict

        Args:
            keys: the keys of the quantity to transfer from image_properties_dict to the spot_properties_dict.
            overwrite: bool, in case of collision between the keys this variable controls
                when the value will be overwritten.
            verbose: bool, if true intermediate messages are displayed.
            strategy: str, either 'closest' (default) or 'bilinear'.
        """

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

        assert isinstance(keys, list), "Error. keys must be a list. Received {0}".format(keys)
        assert set(keys).issubset(set(self.image_properties_dict.keys())), \
            "Some keys are not present in self.image_properties_dict"

        assert strategy == 'bilinear' or strategy == 'closest', "Invalid interpolation_method \
        Expected 'bilinear' or 'closest'. Received {0}. ".format(strategy)

        destination_keys = self._spot_properties_dict.keys()
        for k in keys:
            if k in destination_keys and not overwrite:
                print("The key {0} is already present in spot_properties_dict. \
                        Set overwrite=True to overwrite its value. \
                        Nothing will be done.".format(k))
                return
            if k in destination_keys and overwrite:
                print("The key {0} is already present in spot_properties_dict. \
                        This value will be overwritten".format(k))

        for k in keys:
            if verbose:
                print("working on ->", k)
            image_quantity = self.image_properties_dict[k]

            assert isinstance(image_quantity, torch.Tensor)
            assert len(image_quantity.shape) == 3 and image_quantity.shape[-2:] == self.shape[-2:]

            x_raw = torch.from_numpy(self.x_raw).float()
            y_raw = torch.from_numpy(self.y_raw).float()
            x_pixel, y_pixel = self.raw_to_pixel(x_raw=x_raw, y_raw=y_raw)
            interpolated_values = _interpolation(
                image_quantity, x_pixel, y_pixel, strategy).cpu()

            assert len(interpolated_values.shape) == 2
            self._spot_properties_dict[k] = interpolated_values.permute(dims=(1, 0))

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
            categories_to_codes: dict = None,
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
            categories_to_codes: dictionary with the mapping from the names (of cell_types or genes) to integer codes.
                If not given, the caterogy_values will be inferred from the anndata object.
                Explicitely setting this attribute ensures that the encoding between category
                and integer codes will be consistent across multiple images.
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
            >>>     categories_to_codes=categories_to_codes)
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

            if categories_to_codes is None:
                category_values = list(numpy.unique(cat_raw))
                categories_to_codes = dict(zip(category_values, range(len(category_values))))

            assert set(numpy.unique(cat_raw)).issubset(set(categories_to_codes.keys())), \
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
                categories_to_codes=categories_to_codes,
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

                if verbose:
                    print("working on {0} of shape {1}".format(k, v.shape))

                if len(v.shape) == 1:
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


#    # TODO: add the other methods:
#    #   spot_properties_to_image_properties   -> do it (for discrete you can make a sparse_tensor)
#    #   spot_properties_to_patch properties   -> NO
#    #   patch_properties_to_image_properties  -> implemented
#    #   patch_properties_to_spot_properties   -> NO
#    #   image_properties_to_patch_properties  -> NO
#    #   image_properties_to_spot_properties   -> implemented
#
#    #TODO: Do I need these ones
#    def get_spots_in_patch(self, patch_xywh: Union[torch.Tensor, Tuple[int, int, int, int]]):
#        """ Given a patch, it returns a dictionary with all the spots inside that patch """
#        x_patch, y_patch, w_patch, h_patch = self.__validate_patch_xywh__(patch_xywh)
#        mask = self._are_spots_in_patch(x_patch, y_patch, w_patch, h_patch)
#        return self._get_spots(mask)
#
#    def get_spots_not_in_patch(self, patch_xywh: Union[torch.Tensor, Tuple[int, int, int, int]]):
#        """ Given a patch, it returns a dictionary with all the spots outside that patch """
#        x_patch, y_patch, w_patch, h_patch = self.__validate_patch_xywh__(patch_xywh)
#        mask = self._are_spots_in_patch(x_patch, y_patch, w_patch, h_patch)
#        return self._get_spots(~mask)
#
#    def _are_spots_in_patch(self, x_patch, y_patch, w_patch, h_patch) -> torch.BoolTensor:
#        x_raw = torch.from_numpy(self.x_raw).float()
#        y_raw = torch.from_numpy(self.y_raw).float()
#
#        x_raw_1, y_raw_1 = self.pixel_to_raw(x_patch, y_patch)
#        x_raw_2, y_raw_2 = self.pixel_to_raw(x_patch + w_patch, y_patch + h_patch)
#
#        mask: torch.BoolTensor = (x_raw >= x_raw_1) * (x_raw < x_raw_2) * (y_raw >= y_raw_1) * (y_raw < y_raw_2)
#        return mask
#
#    def _get_spots(self, mask: torch.BoolTensor) -> dict:
#        def _mask_select(_value, _mask):
#            if isinstance(_value, torch.Tensor) or isinstance(_value, numpy.ndarray):
#                return _value[_mask]
#            elif isinstance(_value, list):
#                return numpy.array(_value)[_mask].tolist()
#            else:
#                raise Exception("Expected type Union[torch.Tensor, numpy.ndarray, list]. \
#                Received {0}".format(type(_value)))
#
#        tmp_dict = {}
#        for k, v in self._spot_properties_dict:
#            tmp_dict[k] = _mask_select(v, mask)
#        return tmp_dict
#
#    def create_patches_centered_on_spots(
#            self,
#            spots_strategy: Union[str, torch.BoolTensor],
#            patch_size: Union[int, Tuple[int, int]],
#    ):
#        """
#        Create the patches coordinates (patches_xywh) centered on spots specified by :attr:'spots_strategy.
#
#        Args:
#            spots_strategy: Cab be 'all' or a BoolTensor (i.e. a mask specifying which spots should be a
#                patch associated with them)
#            patch_size: int, the size (in pixel) of the patch
#
#        Returns:
#             None, patches_xywh are written in self.spot_properties_dict
#
#        """
#        assert "patch_xywh" not in self._patch_properties_dict.keys(), "The keyword 'patch_xywh' is already present \
#        in patch_properties_dict. Before you can create new patches you need to call the 'clear_all' method"
#
#        if isinstance(patch_size, int):
#            patch_width = patch_size
#            patch_height = patch_size
#        elif isinstance(patch_size, tuple) and len(patch_size) == 2:
#            patch_width, patch_height = patch_size
#        else:
#            raise Exception("patch_size must be Union[int, Tuple[int,int]]. Received {0}".format(type(patch_size)))
#
#        assert isinstance(patch_height, int) and isinstance(patch_width, int), \
#            "Patch weight and height must be integers. Received {0}, {1}".format(type(patch_width), type(patch_height))
#
#        x_raw = torch.from_numpy(self.x_raw).float()
#        y_raw = torch.from_numpy(self.y_raw).float()
#        x_pixel, y_pixel = self.raw_to_pixel(x_raw, y_raw)
#        x1_pixel = torch.floor(x_pixel - 0.5 * patch_width).int()
#        y1_pixel = torch.floor(y_pixel - 0.5 * patch_height).int()
#        w = patch_width * torch.ones_like(x1_pixel)
#        h = patch_height * torch.ones_like(x1_pixel)
#        patch_xywh = torch.stack((x1_pixel, y1_pixel, w, h), dim=0)
#
#        if spots_strategy == 'all':
#            assert self.n_spots <= 1000, \
#                "Error. The number of requested patches ({0}) is too high.".format(self.n_spots)
#            self._patch_properties_dict["patch_xywh"] = patch_xywh.cpu()
#        elif isinstance(spots_strategy, torch.BoolTensor):
#            assert spots_strategy.shape[0] == patch_xywh.shape[0]
#            self._patch_properties_dict["patch_xywh"] = patch_xywh[spots_strategy].cpu()
#
#    @torch.no_grad()
#    def extract_patches(self,
#                        patches_xywh: torch.Tensor,
#                        cropper: CropperSparseTensor,
#                        transform: Callable = None):
#        """ Return the patches corresponding to the coordinates xywh.
#
#        Args:
#            patches_xywh: torch.tensor of shape (*, 4) with the x,y,w,h coordinates of the patches
#            cropper: a cropper with the :method:'reapply_crops' method
#            transform: a callable which is applied to the sparse tensors to generate a dense tensor. Default is None.
#
#        Returns:
#            A list of patches
#        """
#        # TODO: Can I rewrite this using the equivalent of torch.narrow for sparse data?
#        assert len(patches_xywh.shape) == 2 and patches_xywh.shape[-1] == 4
#        n_patches_max = patches_xywh.shape[0]
#
#        batch_size = 32
#        n_patches = 0
#        all_patches = []
#        while n_patches < n_patches_max:
#            n_added = min(batch_size, n_patches_max - n_patches)
#            crops, x_locs, y_locs = cropper.reapply_crops(self.data, patches_xywh[n_patches:n_patches + n_added])
#            n_patches += n_added
#
#            if transform is not None:
#                patches = transform(crops)
#            else:
#                patches = crops
#
#            all_patches.append(patches.detach().cpu())
#
#        return all_patches
