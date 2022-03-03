import pytorch_lightning as pl

from argparse import ArgumentParser
from argparse import Action as ArgparseAction
import numpy
import os.path
from anndata import read_h5ad
from typing import Dict, Callable, Optional, Tuple, List, Iterable, Any
import torch
import torchvision
from os import cpu_count
from scanpy import AnnData

from tissue_purifier.data_utils.sparse_image import SparseImage
from tissue_purifier.model_utils.patch_analyzer.patch_analyzer import SpatialAutocorrelation
from tissue_purifier.data_utils.transforms import (
    DropoutSparseTensor,
    SparseToDense,
    TransformForList,
    Rasterize,
    RandomHFlip,
    RandomVFlip,
    RandomStraightCut,
    RandomGlobalIntensity,
    DropChannel,
    # LargestSquareCrop,
    # ToRgb,
)
from tissue_purifier.data_utils.dataset import (
    CropperDataset,
    DataLoaderWithLoad,
    CollateFnListTuple,
    MetadataCropperDataset,
    # CropperDenseTensor,
    CropperSparseTensor,
    CropperTensor,
)

# SparseTensor can not be used in dataloader using num_workers > 0.
# See https://github.com/pytorch/pytorch/issues/20248
# Therefore I put the dataset in GPU and use num_workers = 0.


class ParseDict(ArgparseAction):
    """ Make argparse able to parse a dictionary from command line """
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value


class SslDM(pl.LightningDataModule):
    """
    Abstract class to inherit from to make a DataModule which can be used with any
    Self Supervised Learning framework
    """

    @classmethod
    def get_default_params(cls) -> dict:
        parser = ArgumentParser()
        parser = cls.add_specific_args(parser)
        args = parser.parse_args(args=[])
        return args.__dict__

    @classmethod
    def add_specific_args(cls, parent_parser):
        raise NotImplementedError

    def get_metadata_to_regress(self, metadata) -> Dict[str, float]:
        """ Extract one or more quantities to regress from the metadata """
        raise NotImplementedError

    def get_metadata_to_classify(self, metadata) -> Dict[str, int]:
        """ Extract one or more quantities to classify from the metadata """
        raise NotImplementedError

    @property
    def ch_in(self) -> int:
        raise NotImplementedError

    @property
    def local_size(self) -> int:
        """ Size in pixel of the local crops """
        raise NotImplementedError

    @property
    def global_size(self) -> int:
        """ Size in pixel of the global crops """
        raise NotImplementedError

    @property
    def n_local_crops(self) -> int:
        """ Number of local crops for each image to use for training """
        raise NotImplementedError

    @property
    def n_global_crops(self) -> int:
        """ Number of global crops for each image to use for training """
        raise NotImplementedError

    @property
    def cropper_test(self) -> CropperTensor:
        """ Cropper to be used at test time ."""
        raise NotImplementedError

    @property
    def trsfm_test(self) -> Callable:
        """ Transformation to be applied at test time """
        raise NotImplementedError

    @property
    def cropper_train(self) -> CropperTensor:
        """ Cropper to be used at train time."""
        raise NotImplementedError

    @property
    def trsfm_train_local(self) -> Callable:
        """ Local Transformation to be applied at train time """
        raise NotImplementedError

    @property
    def trsfm_train_global(self) -> Callable:
        """ Global Transformation to be applied at train time """
        raise NotImplementedError

    def prepare_data(self):
        # these are things to be done only once in distributed settings
        # good for writing stuff to disk and avoid corruption
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None) -> None:
        # these are things that run on each gpus.
        # Surprisingly, here self.trainer.model.device == cpu
        # while later in dataloader self.trainer.model.device == cuda:0
        # stage: either 'fit', 'validate', 'test', or 'predict'
        raise NotImplementedError

    def train_dataloader(self) -> DataLoaderWithLoad:
        raise NotImplementedError

    def val_dataloader(self) -> List[DataLoaderWithLoad]:
        raise NotImplementedError

    def test_dataloader(self) -> List[DataLoaderWithLoad]:
        raise NotImplementedError

    def predict_dataloader(self) -> List[DataLoaderWithLoad]:
        raise NotImplementedError


class SparseSslDM(SslDM):
    """
    SslDM for sparse Images with the parameter for the transform (i.e. data augmentation) specified.
    If you are inheriting from this class then you only have to overwrite:
    'prepara_data', 'setup', 'get_metadata_to_classify' and 'get_metadata_to_regress'.
    """
    def __init__(self,
                 global_size: int = 96,
                 local_size: int = 64,
                 n_local_crops: int = 2,
                 n_global_crops: int = 2,
                 global_scale: Tuple[float] = (0.8, 1.0),
                 local_scale: Tuple[float] = (0.5, 0.8),
                 global_intensity: Tuple[float, float] = (0.8, 1.2),
                 n_element_min_for_crop: int = 200,
                 dropouts: Tuple[float] = (0.1, 0.2, 0.3),
                 rasterize_sigmas: Tuple[float] = (1.0, 1.5),
                 occlusion_fraction: Tuple[float, float] = (0.1, 0.3),
                 drop_channel_prob: float = 0.0,
                 drop_channel_relative_freq: Iterable[float] = None,
                 n_crops_for_tissue_test: int = 50,
                 n_crops_for_tissue_train: int = 50,
                 # batch_size
                 batch_size_per_gpu: int = 64,
                 **kargs):
        """
        Args:
            global_size: size in pixel of the global crops
            local_size: size in pixel of the local crops
            n_local_crops: number of global crops
            n_global_crops: number of local crops
            global_scale: in RandomResizedCrop the scale of global crops will be drawn uniformly between these values
            local_scale: in RandomResizedCrop the scale of global crops will be drawn uniformly between these values
            global_intensity: all channels will be multiplied by a number in this range
            n_element_min_for_crop: minimum number of beads/cell in a crop
            dropouts: Possible values of the dropout. Should be > 0.0
            rasterize_sigmas: Possible values of the sigma of the gaussian kernel used for rasterization.
            occlusion_fraction: Fraction of the sample which is occluded is drawn uniformly between these values
            drop_channel_prob: Probability that a channel will be set to zero,
            drop_channel_relative_freq: Relative probability of each channel to be set to zero. If None (default) all
                channels are equally likely to be set to zero.
            n_crops_for_tissue_test: The number of crops in each validation epoch will be: n_tissue * n_crops
            n_crops_for_tissue_train: The number of crops in each training epoch will be: n_tissue * n_crops
            batch_size_per_gpu: batch size FOR EACH GPUs.
        """
        super(SparseSslDM, self).__init__()
        
        # params for overwriting the abstract property
        self._global_size = global_size
        self._local_size = local_size
        self._n_global_crops = n_global_crops
        self._n_local_crops = n_local_crops

        # specify the transform
        self._global_scale = global_scale
        self._local_scale = local_scale
        self._global_intensity = global_intensity
        self._dropouts = dropouts
        self._rasterize_sigmas = rasterize_sigmas
        self._occlusion_fraction = occlusion_fraction
        self._drop_channel_prob = drop_channel_prob
        self._drop_channel_relative_freq = drop_channel_relative_freq
        self._n_element_min_for_crop = n_element_min_for_crop
        self._n_crops_for_tissue_test = n_crops_for_tissue_test
        self._n_crops_for_tissue_train = n_crops_for_tissue_train

        # batch_size
        self._batch_size_per_gpu = batch_size_per_gpu
        self.dataset_train: CropperDataset = None
        self.dataset_test: CropperDataset = None

    @classmethod
    def add_specific_args(cls, parent_parser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')

        parser.add_argument("--global_size", type=int, default=96, help="size in pixel of the global crops")
        parser.add_argument("--local_size", type=int, default=64, help="size in pixel of the local crops")
        parser.add_argument("--n_global_crops", type=int, default=2, help="number of global crops")
        parser.add_argument("--n_local_crops", type=int, default=2, help="number of local crops")
        parser.add_argument("--global_scale", type=float, nargs=2, default=[0.8, 1.0],
                            help="in RandomResizedCrop the scale of global crops will be drawn uniformly \
                            between these values")
        parser.add_argument("--local_scale", type=float, nargs=2, default=[0.5, 0.8],
                            help="in RandomResizedCrop the scale of local crops will be drawn uniformly \
                            between these values")
        parser.add_argument("--global_intensity", type=float, nargs=2, default=[0.8, 1.2],
                            help="All channels will be multiplied by a value within this range")
        parser.add_argument("--n_element_min_for_crop", type=int, default=200,
                            help="minimum number of beads/cell in a crop")
        parser.add_argument("--dropouts", type=float, nargs='*', default=[0.1, 0.2, 0.3],
                            help="Possible values of the dropout. Should be > 0.0")
        parser.add_argument("--rasterize_sigmas", type=float, nargs='*', default=[0.5, 1.0, 1.5, 2.0],
                            help="Possible values of the sigma of the gaussian kernel used for rasterization")
        parser.add_argument("--occlusion_fraction", type=float, nargs=2, default=[0.1, 0.3],
                            help="Fraction of the sample which is occluded is drawn uniformly between these values.")
        parser.add_argument("--drop_channel_prob", type=float, default=0.2,
                            help="Probability that a channel in the image will be set to zero.")
        parser.add_argument("--drop_channel_relative_freq", type=float, nargs='*', default=None,
                            help="Relative probability of each channel to be set to zero. \
                            If None, all channels have the same probability of being zero")
        parser.add_argument("--n_crops_for_tissue_train", type=int, default=50,
                            help="The number of crops in each training epoch will be: n_tissue * n_crops. \
                               Set small for rapid prototyping")
        parser.add_argument("--n_crops_for_tissue_test", type=int, default=50,
                            help="The number of crops in each test epoch will be: n_tissue * n_crops. \
                               Set small for rapid prototyping")
        parser.add_argument("--batch_size_per_gpu", type=int, default=64,
                            help="Batch size FOR EACH GPUs. Set small for rapid prototyping. \
                            The total batch_size will increase linearly with the number of GPUs.")
        return parser

    @property
    def global_size(self) -> int:
        return self._global_size

    @property
    def local_size(self) -> int:
        return self._local_size

    @property
    def n_global_crops(self) -> int:
        return self._n_global_crops

    @property
    def n_local_crops(self) -> int:
        return self._n_local_crops

    @property
    def cropper_test(self):
        return CropperSparseTensor(
            strategy='random',
            crop_size=self._global_size,
            n_element_min=self._n_element_min_for_crop,
            n_crops=self._n_crops_for_tissue_test,
            random_order=True,
        )

    @property
    def cropper_train(self):
        return CropperSparseTensor(
            strategy='random',
            crop_size=int(self._global_size * 1.5),
            n_element_min=int(self._n_element_min_for_crop * 1.5 * 1.5),
            n_crops=self._n_crops_for_tissue_train,
            random_order=True,
        )

    @property
    def trsfm_test(self) -> Callable:
        return TransformForList(
            transform_before_stack=torchvision.transforms.Compose([
                DropoutSparseTensor(p=0.5, dropout_rate=self._dropouts),
                SparseToDense(),
                Rasterize(sigmas=self._rasterize_sigmas, normalize=False),
                RandomVFlip(p=0.5),
                RandomHFlip(p=0.5),
                RandomGlobalIntensity(f_min=self._global_intensity[0], f_max=self._global_intensity[1])
            ]),
            transform_after_stack=torchvision.transforms.CenterCrop(size=self.global_size),
        )

    @property
    def trsfm_train_global(self) -> Callable:
        return TransformForList(
            transform_before_stack=torchvision.transforms.Compose([
                DropoutSparseTensor(p=0.5, dropout_rate=self._dropouts),
                SparseToDense(),
                RandomGlobalIntensity(f_min=self._global_intensity[0], f_max=self._global_intensity[1])
            ]),
            transform_after_stack=torchvision.transforms.Compose([
                Rasterize(sigmas=self._rasterize_sigmas, normalize=False),
                torchvision.transforms.RandomRotation(
                    degrees=(-180.0, 180.0),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    expand=False,
                    fill=0.0),
                torchvision.transforms.CenterCrop(size=self._global_size),
                RandomVFlip(p=0.5),
                RandomHFlip(p=0.5),
                torchvision.transforms.RandomResizedCrop(
                    size=(self._global_size, self._global_size),
                    scale=self._global_scale,
                    ratio=(0.95, 1.05),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                RandomStraightCut(p=0.5, occlusion_fraction=self._occlusion_fraction),
                DropChannel(p=self._drop_channel_prob, relative_frequency=self._drop_channel_relative_freq),
            ])
        )

    @property
    def trsfm_train_local(self) -> Callable:
        return TransformForList(
            transform_before_stack=torchvision.transforms.Compose([
                DropoutSparseTensor(p=0.5, dropout_rate=self._dropouts),
                SparseToDense(),
                RandomGlobalIntensity(f_min=self._global_intensity[0], f_max=self._global_intensity[1])
            ]),
            transform_after_stack=torchvision.transforms.Compose([
                Rasterize(sigmas=self._rasterize_sigmas, normalize=False),
                torchvision.transforms.RandomRotation(
                    degrees=(-180.0, 180.0),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    expand=False,
                    fill=0.0),
                torchvision.transforms.CenterCrop(size=self.global_size),
                RandomVFlip(p=0.5),
                RandomHFlip(p=0.5),
                torchvision.transforms.RandomResizedCrop(
                    size=(self._local_size, self._local_size),
                    scale=self._local_scale,
                    ratio=(0.95, 1.05),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                RandomStraightCut(p=0.5, occlusion_fraction=self._occlusion_fraction),
                DropChannel(p=self._drop_channel_prob, relative_frequency=self._drop_channel_relative_freq),
            ])
        )

    def train_dataloader(self) -> DataLoaderWithLoad:
        try:
            device = self.trainer._model.device
        except AttributeError:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # print("Inside train_dataloader", device)

        assert isinstance(self.dataset_train, CropperDataset)
        if self.dataset_train.n_crops_per_tissue is None:
            batch_size_dataloader = self._batch_size_per_gpu
        else:
            batch_size_dataloader = max(1, int(self._batch_size_per_gpu // self.dataset_train.n_crops_per_tissue))

        dataloader_train = DataLoaderWithLoad(
            # move the dataset to GPU so that the cropping happens there
            dataset=self.dataset_train.to(device),
            # each sample generate n_crops therefore reduce batch_size
            batch_size=batch_size_dataloader,
            collate_fn=CollateFnListTuple(),
            # problem if this is larger than 0, see https://github.com/pytorch/pytorch/issues/20248
            num_workers=0,
            # in the train dataloader, I DO shuffle and drop the last partial_batch
            shuffle=True,
            drop_last=True,
        )
        return dataloader_train

    def val_dataloader(self) -> List[DataLoaderWithLoad]:  # the same as test
        try:
            device = self.trainer._model.device
        except AttributeError:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        assert isinstance(self.dataset_test, CropperDataset)
        if self.dataset_test.n_crops_per_tissue is None:
            batch_size_dataloader = self._batch_size_per_gpu
        else:
            batch_size_dataloader = max(1, int(self._batch_size_per_gpu // self.dataset_train.n_crops_per_tissue))

        assert isinstance(self.dataset_test, CropperDataset)
        test_dataloader = DataLoaderWithLoad(
            # move the dataset to GPU so that the cropping happens there
            dataset=self.dataset_test.to(device),
            # each sample generate n_crops therefore reduce batch_size
            batch_size=batch_size_dataloader,
            collate_fn=CollateFnListTuple(),
            # problem if num_workers > 0, see https://github.com/pytorch/pytorch/issues/20248
            num_workers=0,
            # in the test dataloader, I do NOT shuffle and do not drop the last partial_batch
            shuffle=False,
            drop_last=False,
        )
        return [test_dataloader]

    def test_dataloader(self) -> List[DataLoaderWithLoad]:
        return self.val_dataloader()

    def predict_dataloader(self) -> List[DataLoaderWithLoad]:
        return self.val_dataloader()

    def prepare_data(self):
        raise NotImplementedError

    def get_metadata_to_classify(self, metadata) -> Dict[str, int]:
        raise NotImplementedError

    def get_metadata_to_regress(self, metadata) -> Dict[str, float]:
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset_train = None
        self.dataset_test = None
        raise NotImplementedError


class AnndataFolderDM(SparseSslDM):
    """
    Create a Datamodule ready for Self-supervised learning starting
    from a folder full of anndata file in h5ad format.
    """
    def __init__(self,
                 data_folder: str,
                 pixel_size: float,
                 x_key: str,
                 y_key: str,
                 category_key: str,
                 categories_to_channels: Dict[Any, int],
                 metadata_to_classify: Callable,
                 metadata_to_regress: Callable,
                 num_workers: int,
                 gpus: int,
                 n_neighbours_moran: int,
                 **kargs):

        self._data_folder = data_folder
        self._pixel_size = pixel_size
        self._x_key = x_key
        self._y_key = y_key
        self._category_key = category_key
        self._categories_to_channels = categories_to_channels
        self._metadata_to_regress = metadata_to_regress
        self._metadata_to_classify = metadata_to_classify

        self._num_workers = cpu_count() if num_workers is None else num_workers
        self._gpus = torch.cuda.device_count() if gpus is None else gpus
        self._n_neighbours_moran = n_neighbours_moran

        # Callable on dataset
        self.compute_moran = SpatialAutocorrelation(
            modality='moran',
            n_neighbours=self._n_neighbours_moran,
            neigh_correct=False)

        # list of all the files used to create the dataset
        self._all_filenames = None

        super(AnndataFolderDM, self).__init__(**kargs)

    @classmethod
    def add_specific_args(cls, parent_parser) -> ArgumentParser:
        parser_from_super = super().add_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser_from_super], add_help=False, conflict_handler='resolve')

        parser.add_argument("--data_folder", type=str, default="./",
                            help="directory where to find the anndata in h5ad format")
        parser.add_argument("--pixel_size", type=float, default=4.0,
                            help="size of the pixel (used to convert raw_coordinates to pixel_coordinates)")
        parser.add_argument("--x_key", type=str, default="x",
                            help="key associated with the x_coordinate in the AnnData object")
        parser.add_argument("--y_key", type=str, default="y",
                            help="key associated with the y_coordinate in the AnnData object")
        parser.add_argument("--category_key", type=str, default="cell_type",
                            help="key associated with the the categorical values (cell_types or gene_identities) \
                            in the AnnData object")
        parser.add_argument("--categories_to_channels", nargs='*', action=ParseDict,
                            help="dictionary in the form 'foo'=1 'bar'=2 to define \
                            how the categorical values are mapped to the different channels in the image")
        parser.add_argument("--metadata_to_classify", default=None,
                            help="callable which defines the values to classify during training")
        parser.add_argument("--metadata_to_regress", default=None,
                            help="callable which defines the values to regress during training")
        parser.add_argument("--num_workers", default=cpu_count(), type=int,
                            help="number of worker to load data. Meaningful only if dataset is on disk. \
                            Set to zero if data in memory")
        parser.add_argument("--gpus", default=None, type=int,
                            help="number of gpus to use for training. If None (default) I uses all the gpus available")
        parser.add_argument("--n_neighbours_moran", type=int, default=6,
                            help="number of neighbours used to compute moran")
        return parser

    @property
    def ch_in(self) -> int:
        return numpy.max(list(self._categories_to_channels.values())) + 1

    def anndata_to_sparseimage(self, anndata: AnnData):
        """ Method which converts a anndata object to sparse image """
        return SparseImage.from_anndata(
            anndata=anndata,
            x_key=self._x_key,
            y_key=self._y_key,
            category_key=self._category_key,
            pixel_size=self._pixel_size,
            categories_to_channels=self._categories_to_channels,
            padding=10)

    def prepare_data(self):
        # these are things to be done only once in distributed settings
        # good for writing stuff to disk and avoid corruption

        # create train_dataset_random and write to file
        all_metadatas = []
        all_sparse_images = []
        all_labels = []

        for filename in os.listdir(self._data_folder):
            f = os.path.join(self._data_folder, filename)
            # checking if it is a file
            if os.path.isfile(f) and filename.endswith('h5ad'):
                print("reading file {}".format(f))
                anndata = read_h5ad(filename=f)
                anndata.X = None  # set the count matrix to None
                sp_img = self.anndata_to_sparseimage(anndata=anndata).cpu()
                all_sparse_images.append(sp_img)

                metadata = MetadataCropperDataset(f_name=filename, loc_x=0.0, loc_y=0.0, moran=-99)
                all_metadatas.append(metadata)

                all_labels.append(filename)

        self._all_filenames: list = all_labels

        torch.save((all_sparse_images, all_labels, all_metadatas),
                   os.path.join(self._data_folder, "train_dataset.pt"))
        print("saved the file", os.path.join(self._data_folder, "train_dataset.pt"))

        # create test_dataset_random and write to file
        all_names = [metadata.f_name for metadata in all_metadatas]

        if torch.cuda.is_available():
            all_sparse_images = [sp_img.cuda() for sp_img in all_sparse_images]

        test_imgs, test_labels, test_metadatas = [], [], []
        for sp_img, label, fname in zip(all_sparse_images, all_labels, all_names):
            sps_tmp, loc_x_tmp, loc_y_tmp = self.cropper_test(sp_img, n_crops=self._n_crops_for_tissue_test)
            labels = [label] * len(sps_tmp)

            morans = [self.compute_moran(sparse_tensor).max().item() for sparse_tensor in sps_tmp]
            metadatas = [MetadataCropperDataset(f_name=fname, loc_x=loc_x, loc_y=loc_y, moran=moran) for
                         loc_x, loc_y, moran in zip(loc_x_tmp, loc_y_tmp, morans)]

            test_imgs += [sp_img.cpu() for sp_img in sps_tmp]
            test_labels += labels
            test_metadatas += metadatas

        torch.save((test_imgs, test_labels, test_metadatas), os.path.join(self._data_folder, "test_dataset.pt"))
        print("saved the file", os.path.join(self._data_folder, "test_dataset.pt"))

    def get_metadata_to_classify(self, metadata) -> Dict[str, int]:
        if self._metadata_to_classify is None:
            return {"tissue_label": self._all_filenames.index(metadata.f_name)}
        else:
            return self._metadata_to_classify(metadata)

    def get_metadata_to_regress(self, metadata) -> Dict[str, float]:
        if self._metadata_to_regress is None:
            return {
                "moran": float(metadata.moran),
                "loc_x": float(metadata.loc_x),
            }
        else:
            return self._metadata_to_regress(metadata)

    def setup(self, stage: Optional[str] = None) -> None:
        list_imgs, list_labels, list_metadata = torch.load(os.path.join(self._data_folder, "train_dataset.pt"))
        list_imgs = [img.coalesce().cpu() for img in list_imgs]
        self.dataset_train = CropperDataset(
            imgs=list_imgs,
            labels=list_labels,
            metadatas=list_metadata,
            cropper=self.cropper_train,
        )
        print("created train_dataset device = {0}, length = {1}".format(self.dataset_train.imgs[0].device,
                                                                        self.dataset_train.__len__()))

        list_imgs, list_labels, list_metadata = torch.load(os.path.join(self._data_folder, "test_dataset.pt"))
        list_imgs = [img.coalesce().cpu() for img in list_imgs]
        self.dataset_test = CropperDataset(
            imgs=list_imgs,
            labels=list_labels,
            metadatas=list_metadata,
            cropper=None,
        )
        print("created test_dataset device = {0}, length = {1}".format(self.dataset_test.imgs[0].device,
                                                                       self.dataset_test.__len__()))
