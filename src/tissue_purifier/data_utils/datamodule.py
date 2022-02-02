import pytorch_lightning as pl

from argparse import ArgumentParser
from os import listdir
import tarfile
import pandas as pd
import os.path
from anndata import read_h5ad
from typing import Dict, Callable, Optional, Tuple, List
import torch
import torchvision
from os import cpu_count

from tissue_purifier.misc_utils.misc import smart_bool

from tissue_purifier.data_utils.sparse_image import SparseImage
from tissue_purifier.model_utils.analyzer import SpatialAutocorrelation
from tissue_purifier.data_utils.transforms import (
    DropoutSparseTensor,
    SparseToDense,
    TransformForList,
    Rasterize,
    RandomHFlip,
    RandomVFlip,
    RandomStraightCut,
    RandomGlobalIntensity,
    # LargestSquareCrop,
    # ToRgb,
)
from tissue_purifier.data_utils.dataset import (
    # AddFakeMetadata,
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


class classproperty(property):
    """ Own-made decorator for defining cls properties """
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


class DinoDM(pl.LightningDataModule):
    """ Abstract class to inherit from to make a dataset to be used with the DINO model """

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

    @classproperty
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


class DinoSparseDM(DinoDM):
    """
    DinoDM for sparse Images with the parameter for the transform specified.
    If you are inheriting from this class then you have to overwrite:
    'prepara_data', 'setup', 'get_metadata_to_classify' and 'get_metadata_to_regress'.
    """
    def __init__(self,
                 global_size: int = 96,
                 local_size: int = 64,
                 n_local_crops: int = 2,
                 n_global_crops: int = 2,
                 global_scale: Tuple[float] = (0.8, 1.0),
                 local_scale: Tuple[float] = (0.5, 0.8),
                 n_element_min_for_crop: int = 200,
                 dropouts: Tuple[float] = (0.1, 0.2, 0.3),
                 rasterize_sigmas: Tuple[float] = (0.5, 1.0, 1.5, 2.0),
                 occlusion_fraction: Tuple[float, float] = (0.1, 0.3),
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
            n_element_min_for_crop: minimum number of beads/cell in a crop
            dropouts: Possible values of the dropout. Should be > 0.0
            rasterize_sigmas: Possible values of the sigma of the gaussian kernel used for rasterization.
            occlusion_fraction: Fraction of the sample which is occluded is drawn uniformly between these values
            n_crops_for_tissue_test: The number of crops in each validation epoch will be: n_tissue * n_crops
            n_crops_for_tissue_train: The number of crops in each training epoch will be: n_tissue * n_crops
            batch_size_per_gpu: batch size FOR EACH GPUs.
        """
        super(DinoSparseDM, self).__init__()
        
        # params for overwriting the abstract property
        self._global_size = global_size
        self._local_size = local_size
        self._n_global_crops = n_global_crops
        self._n_local_crops = n_local_crops

        # specify the transform
        self._global_scale = global_scale
        self._local_scale = local_scale
        self._dropouts = dropouts
        self._rasterize_sigmas = rasterize_sigmas
        self._occlusion_fraction = occlusion_fraction
        self._n_element_min_for_crop = n_element_min_for_crop
        self._n_crops_for_tissue_test = n_crops_for_tissue_test
        self._n_crops_for_tissue_train = n_crops_for_tissue_train

        # batch_size
        self._batch_size_per_gpu = batch_size_per_gpu
        self.dataset_train = None
        self.dataset_test = None

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
        parser.add_argument("--n_element_min_for_crop", type=int, default=200,
                            help="minimum number of beads/cell in a crop")
        parser.add_argument("--dropouts", type=float, nargs='*', default=[0.1, 0.2, 0.3],
                            help="Possible values of the dropout. Should be > 0.0")
        parser.add_argument("--rasterize_sigmas", type=float, nargs='*', default=[0.5, 1.0, 1.5, 2.0],
                            help="Possible values of the sigma of the gaussian kernel used for rasterization")
        parser.add_argument("--occlusion_fraction", type=float, nargs=2, default=[0.1, 0.3],
                            help="Fraction of the sample which is occluded is drawn uniformly between these values.")
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
                DropoutSparseTensor(p=0.5, dropouts=self._dropouts),
                SparseToDense(),
                Rasterize(sigmas=self._rasterize_sigmas, normalize=False),
                RandomVFlip(p=0.5),
                RandomHFlip(p=0.5),
                RandomGlobalIntensity(f_min=0.8, f_max=1.2)
            ]),
            transform_after_stack=torchvision.transforms.CenterCrop(size=self.global_size),
        )

    @property
    def trsfm_train_global(self) -> Callable:
        return TransformForList(
            transform_before_stack=torchvision.transforms.Compose([
                DropoutSparseTensor(p=0.5, dropouts=self._dropouts),
                SparseToDense(),
                RandomGlobalIntensity(f_min=0.8, f_max=1.2)
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
            ])
        )

    @property
    def trsfm_train_local(self) -> Callable:
        return TransformForList(
            transform_before_stack=torchvision.transforms.Compose([
                DropoutSparseTensor(p=0.5, dropouts=self._dropouts),
                SparseToDense(),
                RandomGlobalIntensity(f_min=0.8, f_max=1.2)
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
            ])
        )

    def train_dataloader(self) -> DataLoaderWithLoad:
        try:
            device = self.trainer.model.device
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
            device = self.trainer.model.device
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

    def setup(self, stage: Optional[str] = None) -> None:
        """ Must overwrite this function and redefine dataset_train, dataset_test """
        self.dataset_train = None
        self.dataset_test = None
        raise NotImplementedError


class DummyDM(DinoSparseDM):
    def __init__(self, **kargs):
        """ All inputs are neglected and the default values are applied. """
        print("-----> running datamodule init")

        configs_for_transforms = {
            'batch_size_per_gpu': 10,
            'n_crops_for_tissue_test': 10,
            'n_crops_for_tissue_train': 10,
            'n_element_min_for_crop': 1,
            'global_size': 64,
            'global_scale': (0.75, 1.0),
            'local_size': 32,
            'local_scale': (0.5, 0.75)
        }
        for k,v in configs_for_transforms.items():
            kargs[k] = v

        super(DummyDM, self).__init__(**kargs)

        self._data_dir = "./dummy_data"
        self._num_workers = 1
        self._gpus = torch.cuda.device_count()
        self._load_count_matrix = False
        self._pixel_size = 1.0
        self._n_neighbours_moran = 6

        # Callable on dataset
        self.compute_moran = SpatialAutocorrelation(
            modality='moran',
            n_neighbours=self._n_neighbours_moran,
            neigh_correct=False)
        self.dataset_train = None
        self.dataset_test = None

    @classmethod
    def add_specific_args(cls, parent_parser) -> ArgumentParser:
        parser_from_super = super().add_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser_from_super], add_help=False, conflict_handler='resolve')

        parser.add_argument("--data_dir", type=str, default="./slide_seq_testis",
                            help="directory where to download the data")
        parser.add_argument("--num_workers", default=cpu_count(), type=int,
                            help="number of worker to load data. Meaningful only if dataset is on disk. \
                            Set to zero if data in memory")
        parser.add_argument("--pixel_size", type=float, default=4.0,
                            help="size of the pixel (used to convert raw_coordinates to pixel_coordinates)")
        parser.add_argument("--load_count_matrix", type=smart_bool, default=False,
                            help="If true load the count matrix in the anndata object. \
                            Count matrix is memory intensive therefore it can be advantegeous not to load it.")
        parser.add_argument("--n_neighbours_moran", type=int, default=6,
                            help="number of neighbours used to compute moran")

        return parser

    def get_metadata_to_regress(self, metadata: MetadataCropperDataset) -> Dict[str, float]:
        """ Extract one or more quantities to regress from the metadata """
        return {
            "moran": float(metadata.moran),
            "loc_x": float(metadata.loc_x),
        }

    def get_metadata_to_classify(self, metadata: MetadataCropperDataset) -> Dict[str, int]:
        """
        Extract one or more quantities to classify from the metadata.
        One should be at "tissue_label" b/c it is used to select non-overlapping patches.
        """

        def _remove_prefix(x):
            return int(x.lstrip("id_"))

        return {
            "tissue_label": _remove_prefix(metadata.f_name)
        }

    @classproperty
    def ch_in(self) -> int:
        return 9

    def prepare_data(self):
        cell_list = ["ES", "Endothelial", "Leydig", "Macrophage", "Myoid", "RS", "SPC", "SPG", "Sertoli"]
        categories_to_codes = dict(zip(cell_list, range(len(cell_list))))

        import random
        import pandas
        import numpy
        from anndata import AnnData

        n_tissues, n_beads = 5, 5000
        all_anndata, all_names_sparse_images, all_labels_sparse_images = [], [], []
        for n_tissue in range(n_tissues):
            tmp_dict = {
                "x_raw": 200.0 + 500.0 * numpy.random.rand(n_beads),
                "y_raw": -200.0 + 500.0 * numpy.random.rand(n_beads),
                "cell_type": [cell_list[i] for i in numpy.random.randint(low=0, high=9, size=n_beads)],
                "barcodes": [random.getrandbits(128) for i in range(n_beads)]
            }
            metadata_df = pandas.DataFrame(data=tmp_dict).set_index("barcodes")
            adata = AnnData(obs=metadata_df)

            all_anndata.append(adata)
            all_names_sparse_images.append("id_"+str(n_tissue))
            all_labels_sparse_images.append("wt" if numpy.random.rand() < 0.5 else "dis")

        all_metadata = [MetadataCropperDataset(f_name=f_name, loc_x=0.0, loc_y=0.0, moran=-99.9) for
                        f_name in all_names_sparse_images]

        # create the train_dataset and write to file
        all_sparse_images = [SparseImage.from_anndata(
            anndata,
            x_key="x_raw",
            y_key="y_raw",
            category_key="cell_type",
            pixel_size=self._pixel_size,
            padding=10,
            categories_to_codes=categories_to_codes) for anndata in all_anndata]

        all_sparse_images_cpu = [sp_image.cpu() for sp_image in all_sparse_images]
        os.makedirs(self._data_dir, exist_ok=True)
        torch.save((all_sparse_images_cpu, all_labels_sparse_images, all_metadata),
                   os.path.join(self._data_dir, "train_dataset.pt"))
        print("saved the file", os.path.join(self._data_dir, "train_dataset.pt"))

        # create test_dataset_random and write to file
        list_imgs, list_labels, list_fnames, list_loc_xs, list_loc_ys, list_morans = [], [], [], [], [], []
        for sp_img, label, fname in zip(all_sparse_images, all_labels_sparse_images, all_names_sparse_images):
            sps_tmp, loc_x_tmp, loc_y_tmp = self.cropper_test(sp_img, n_crops=self._n_crops_for_tissue_test)
            list_morans += [self.compute_moran(sparse_tensor).max().item() for sparse_tensor in sps_tmp]
            list_imgs += sps_tmp
            list_labels += [label] * len(sps_tmp)
            list_fnames += [fname] * len(sps_tmp)
            list_loc_xs += loc_x_tmp
            list_loc_ys += loc_y_tmp

        list_metadata = [MetadataCropperDataset(f_name=f_name, loc_x=loc_x, loc_y=loc_y, moran=moran) for
                         f_name, loc_x, loc_y, moran in zip(list_fnames, list_loc_xs, list_loc_ys, list_morans)]
        list_imgs_cpu = [img.cpu() for img in list_imgs]
        os.makedirs(self._data_dir, exist_ok=True)
        torch.save((list_imgs_cpu, list_labels, list_metadata), os.path.join(self._data_dir, "test_dataset.pt"))
        print("saved the file", os.path.join(self._data_dir, "test_dataset.pt"))

    def setup(self, stage: Optional[str] = None) -> None:
        list_imgs, list_labels, list_metadata = torch.load(os.path.join(self._data_dir, "train_dataset.pt"))
        list_imgs = [img.coalesce().cpu() for img in list_imgs]
        self.dataset_train = CropperDataset(
            imgs=list_imgs,
            labels=list_labels,
            metadatas=list_metadata,
            cropper=self.cropper_train,
        )
        print("created train_dataset device = {0}, length = {1}".format(self.dataset_train.imgs[0].device,
                                                                        self.dataset_train.__len__()))

        list_imgs, list_labels, list_metadata = torch.load(os.path.join(self._data_dir, "test_dataset.pt"))
        list_imgs = [img.coalesce().cpu() for img in list_imgs]
        self.dataset_test = CropperDataset(
            imgs=list_imgs,
            labels=list_labels,
            metadatas=list_metadata,
            cropper=None,
        )
        print("created test_dataset device = {0}, length = {1}".format(self.dataset_test.imgs[0].device,
                                                                       self.dataset_test.__len__()))


class SlideSeqTestisDM(DinoSparseDM):
    def __init__(self,
                 data_dir: str = './slide_seq_testis',
                 num_workers: int = None,
                 gpus: int = None,
                 pixel_size: int = 4,
                 load_count_matrix: bool = False,
                 n_neighbours_moran: int = 6,
                 **kargs):
        # new_params
        self._data_dir = data_dir
        self._num_workers = cpu_count() if num_workers is None else num_workers
        self._gpus = torch.cuda.device_count() if gpus is None else gpus
        self._pixel_size = pixel_size
        self._load_count_matrix = load_count_matrix
        self._n_neighbours_moran = n_neighbours_moran

        # Callable on dataset
        self.compute_moran = SpatialAutocorrelation(
            modality='moran',
            n_neighbours=self._n_neighbours_moran,
            neigh_correct=False)

        super(SlideSeqTestisDM, self).__init__(**kargs)
        print("-----> running datamodule init")

    @classmethod
    def add_specific_args(cls, parent_parser) -> ArgumentParser:
        parser_from_super = super().add_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser_from_super], add_help=False, conflict_handler='resolve')

        parser.add_argument("--data_dir", type=str, default="./slide_seq_testis",
                            help="directory where to download the data")
        parser.add_argument("--num_workers", default=cpu_count(), type=int,
                            help="number of worker to load data. Meaningful only if dataset is on disk. \
                            Set to zero if data in memory")
        parser.add_argument("--pixel_size", type=float, default=4.0,
                            help="size of the pixel (used to convert raw_coordinates to pixel_coordinates)")
        parser.add_argument("--load_count_matrix", type=smart_bool, default=False,
                            help="If true load the count matrix in the anndata object. \
                            Count matrix is memory intensive therefore it can be advantegeous not to load it.")
        parser.add_argument("--n_neighbours_moran", type=int, default=6,
                            help="number of neighbours used to compute moran")

        return parser

    def get_metadata_to_regress(self, metadata: MetadataCropperDataset) -> Dict[str, float]:
        """ Extract one or more quantities to regress from the metadata """
        return {
            "moran": float(metadata.moran),
            "loc_x": float(metadata.loc_x),
        }

    def get_metadata_to_classify(self, metadata: MetadataCropperDataset) -> Dict[str, int]:
        """ Extract one or more quantities to classify from the metadata """
        conversion1 = {
            'wt1': 0,
            'wt2': 1,
            'wt3': 2,
            'dis1': 3,
            'dis2': 4,
            'dis3': 5,
        }

        conversion2 = {
            'wt1': 0,
            'wt2': 0,
            'wt3': 0,
            'dis1': 1,
            'dis2': 1,
            'dis3': 1,
        }

        return {
            "tissue_label": conversion1[metadata.f_name],
            "healthy_sick": conversion2[metadata.f_name],
        }

    @classproperty
    def ch_in(self) -> int:
        return 9

    def prepare_data(self):
        # these are things to be done only once in distributed settings
        # good for writing stuff to disk and avoid corruption
        print("-----> running datamodule prepare_data")

        def __tar_exists__():
            return os.path.exists(os.path.join(self._data_dir, "slideseq_testis_anndata_h5ad.tar.gz"))

        def __download__tar__():
            from google.cloud import storage

            bucket_name = "ld-data-bucket"
            source_blob_name = "tissue-purifier/slideseq_testis_anndata_h5ad.tar.gz"
            destination_file_name = os.path.join(self._data_dir, "slideseq_testis_anndata_h5ad.tar.gz")

            # create the directory where the file will be written
            dirname_tmp = os.path.dirname(destination_file_name)
            os.makedirs(dirname_tmp, exist_ok=True)

            # connect ot the google bucket
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)

            # Construct a client side representation of a blob.
            # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
            # any content from Google Cloud Storage. As we don't need additional data,
            # using `Bucket.blob` is preferred here.
            blob = bucket.blob(source_blob_name)
            blob.download_to_filename(destination_file_name)

            print(
                "Downloaded storage object {} from bucket {} to local file {}.".format(
                    source_blob_name, bucket_name, destination_file_name
                )
            )

        def __untar__():
            tar_file_name = os.path.join(self._data_dir, "slideseq_testis_anndata_h5ad.tar.gz")

            # untar the tar.gz file
            with tarfile.open(tar_file_name, "r:gz") as fp:
                fp.extractall(path=self._data_dir)

        print("Will create the test and train file")
        if __tar_exists__():
            print("untar data")
            __untar__()
        else:
            print("download and untar data")
            __download__tar__()
            __untar__()

        anndata_wt1 = read_h5ad(os.path.join(self._data_dir, "anndata_wt1.h5ad"))
        anndata_wt2 = read_h5ad(os.path.join(self._data_dir, "anndata_wt2.h5ad"))
        anndata_wt3 = read_h5ad(os.path.join(self._data_dir, "anndata_wt3.h5ad"))
        anndata_sick1 = read_h5ad(os.path.join(self._data_dir, "anndata_sick1.h5ad"))
        anndata_sick2 = read_h5ad(os.path.join(self._data_dir, "anndata_sick2.h5ad"))
        anndata_sick3 = read_h5ad(os.path.join(self._data_dir, "anndata_sick3.h5ad"))

        # create train_dataset and write to file
        cell_list = ["ES", "Endothelial", "Leydig", "Macrophage", "Myoid", "RS", "SPC", "SPG", "Sertoli"]
        categories_to_codes = dict(zip(cell_list, range(len(cell_list))))
        all_anndata = [anndata_wt1, anndata_wt2, anndata_wt3, anndata_sick1, anndata_sick2, anndata_sick3]
        all_labels_sparse_images = ['wt', 'wt', 'wt', 'sick', 'sick', 'sick']
        all_names_sparse_images = ["wt1", "wt2", "wt3", "dis1", "dis2", "dis3"]
        all_metadata = [MetadataCropperDataset(f_name=f_name, loc_x=0.0, loc_y=0.0, moran=-99.9) for
                        f_name in all_names_sparse_images]

        # set the count matrix to None (if necessary)
        if not self._load_count_matrix:
            for anndata in all_anndata:
                anndata.X = None

        # create the train_dataset and write to file
        all_sparse_images = [SparseImage.from_anndata(
            anndata,
            x_key="x",
            y_key="y",
            category_key="cell_type",
            pixel_size=self._pixel_size,
            padding=10,
            categories_to_codes=categories_to_codes) for anndata in all_anndata]

        all_sparse_images_cpu = [sp_image.cpu() for sp_image in all_sparse_images]
        torch.save((all_sparse_images_cpu, all_labels_sparse_images, all_metadata),
                   os.path.join(self._data_dir, "train_dataset.pt"))
        print("saved the file", os.path.join(self._data_dir, "train_dataset.pt"))

        # create test_dataset_random and write to file
        list_imgs, list_labels, list_fnames, list_loc_xs, list_loc_ys, list_morans = [], [], [], [], [], []
        for sp_img, label, fname in zip(all_sparse_images, all_labels_sparse_images, all_names_sparse_images):
            sps_tmp, loc_x_tmp, loc_y_tmp = self.cropper_test(sp_img, n_crops=self._n_crops_for_tissue_test)
            list_morans += [self.compute_moran(sparse_tensor).max().item() for sparse_tensor in sps_tmp]
            list_imgs += sps_tmp
            list_labels += [label] * len(sps_tmp)
            list_fnames += [fname] * len(sps_tmp)
            list_loc_xs += loc_x_tmp
            list_loc_ys += loc_y_tmp

        list_metadata = [MetadataCropperDataset(f_name=f_name, loc_x=loc_x, loc_y=loc_y, moran=moran) for
                         f_name, loc_x, loc_y, moran in zip(list_fnames, list_loc_xs, list_loc_ys, list_morans)]
        list_imgs_cpu = [img.cpu() for img in list_imgs]
        torch.save((list_imgs_cpu, list_labels, list_metadata), os.path.join(self._data_dir, "test_dataset.pt"))
        print("saved the file", os.path.join(self._data_dir, "test_dataset.pt"))

    def setup(self, stage: Optional[str] = None) -> None:
        list_imgs, list_labels, list_metadata = torch.load(os.path.join(self._data_dir, "train_dataset.pt"))
        list_imgs = [img.coalesce().cpu() for img in list_imgs]
        self.dataset_train = CropperDataset(
            imgs=list_imgs,
            labels=list_labels,
            metadatas=list_metadata,
            cropper=self.cropper_train,
        )
        print("created train_dataset device = {0}, length = {1}".format(self.dataset_train.imgs[0].device,
                                                                        self.dataset_train.__len__()))

        list_imgs, list_labels, list_metadata = torch.load(os.path.join(self._data_dir, "test_dataset.pt"))
        list_imgs = [img.coalesce().cpu() for img in list_imgs]
        self.dataset_test = CropperDataset(
            imgs=list_imgs,
            labels=list_labels,
            metadatas=list_metadata,
            cropper=None,
        )
        print("created test_dataset device = {0}, length = {1}".format(self.dataset_test.imgs[0].device,
                                                                       self.dataset_test.__len__()))


class SlideSeqKidneyDM(DinoSparseDM):
    def __init__(self,
                 cohort: str = 'all',
                 data_dir: str = './slide_seq_kidney',
                 num_workers: int = None,
                 gpus: int = None,
                 pixel_size: int = 4,
                 load_count_matrix: bool = False,
                 n_neighbours_moran: int = 6,
                 **kargs):
        # new_params
        self._cohort = cohort
        self._data_dir = data_dir
        self._num_workers = cpu_count() if num_workers is None else num_workers
        self._gpus = torch.cuda.device_count() if gpus is None else gpus
        self._pixel_size = pixel_size
        self._load_count_matrix = load_count_matrix
        self._n_neighbours_moran = n_neighbours_moran

        # Callable on dataset
        self.compute_moran = SpatialAutocorrelation(
            modality='moran',
            n_neighbours=self._n_neighbours_moran,
            neigh_correct=False)

        # dictionary which will be created during prepare_data
        self.array_to_code = None
        self.array_to_species = None
        self.array_to_condition = None
        super(SlideSeqKidneyDM, self).__init__(**kargs)
        print("-----> running datamodule init")

    @classmethod
    def add_specific_args(cls, parent_parser) -> ArgumentParser:
        parser_from_super = super().add_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser_from_super], add_help=False, conflict_handler='resolve')
        parser.add_argument("--cohort", default='test', type=str,
                            choices=['small', 'all', 'mouse_only', 'human_only'],
                            help="Specify grouping of samples to use during training and testing")
        parser.add_argument("--data_dir", type=str, default="./slide_seq_testis",
                            help="directory where to download the data")
        parser.add_argument("--num_workers", default=cpu_count(), type=int,
                            help="number of worker to load data. Meaningful only if dataset is on disk. \
                            Set to zero if data in memory")
        parser.add_argument("--pixel_size", type=float, default=4.0,
                            help="size of the pixel (used to convert raw_coordinates to pixel_coordinates)")
        parser.add_argument("--load_count_matrix", type=smart_bool, default=False,
                            help="If true load the count matrix in the anndata object. \
                            Count matrix is memory intensive therefore it can be advantegeous not to load it.")
        parser.add_argument("--n_neighbours_moran", type=int, default=6,
                            help="number of neighbours used to compute moran")

        return parser

    def get_metadata_to_regress(self, metadata: MetadataCropperDataset) -> Dict[str, float]:
        """ Extract one or more quantities to regress from the metadata """
        return {
            "moran": float(metadata.moran),
            "loc_x": float(metadata.loc_x),
        }

    def get_metadata_to_classify(self, metadata: MetadataCropperDataset) -> Dict[str, int]:
        """ Extract one or more quantities to classify from the metadata """

        def species_to_code(specie):
            if specie == 'human':
                return 0
            elif specie == 'mouse':
                return 1
            else:
                return 2

        def condition_to_code(sample):
            if sample.startswith('UMOD'):
                return 0
            elif sample.startswith('WT'):
                return 1
            elif sample.startswith('DKD'):
                return 2
            elif sample.endswith('cortex'):
                return 3
            elif sample.endswith('med'):
                return 4
            else:
                return 5

        return {
            "tissue_label": self.array_to_code[metadata.f_name],
            "species_code": species_to_code(self.array_to_species[metadata.f_name]),
            "condition_code": condition_to_code(self.array_to_condition[metadata.f_name])
        }

    @classproperty
    def ch_in(self) -> int:
        return 13

    def prepare_data(self):
        # these are things to be done only once in distributed settings
        print("-----> running datamodule prepare_data")

        def __tar_exists__():
            return os.path.exists(os.path.join(self._data_dir, "slideseq_kidney_anndata_h5ad.tar.gz"))

        def __download__tar__():
            from google.cloud import storage

            bucket_name = "ld-data-bucket"
            source_blob_name = "tissue-purifier/slideseq_kidney_anndata_h5ad.tar.gz"
            destination_file_name = os.path.join(self._data_dir, "slideseq_kidney_anndata_h5ad.tar.gz")

            # create the directory where the file will be written
            dirname_tmp = os.path.dirname(destination_file_name)
            os.makedirs(dirname_tmp, exist_ok=True)

            # connect ot the google bucket
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)

            # Construct a client side representation of a blob.
            # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
            # any content from Google Cloud Storage. As we don't need additional data,
            # using `Bucket.blob` is preferred here.
            blob = bucket.blob(source_blob_name)
            blob.download_to_filename(destination_file_name)

            print(
                "Downloaded storage object {} from bucket {} to local file {}.".format(
                    source_blob_name, bucket_name, destination_file_name
                )
            )

        def __untar__():
            tar_file_name = os.path.join(self._data_dir, "slideseq_kidney_anndata_h5ad.tar.gz")

            # untar the tar.gz file
            with tarfile.open(tar_file_name, "r:gz") as fp:
                fp.extractall(path=self._data_dir)

        print("Will create the test and train file")
        if __tar_exists__():
            print("untar data")
            __untar__()
        else:
            print("download and untar data")
            __download__tar__()
            __untar__()

        df_archive = pd.read_csv(os.path.join(self._data_dir, "slideseq_kidney_datatable.csv"),
                                 usecols=["arrays", "samples", "species"])
        array_list = df_archive["arrays"].tolist()
        condition_list = df_archive["samples"].tolist()
        array_to_condition = dict(zip(array_list, condition_list))

        print("COHORT -->", self._cohort)
        if self._cohort == 'small':
            puck_id_list = df_archive.head(6)['arrays'].tolist()
        elif self._cohort == 'mouse_only':
            puck_id_list = df_archive[df_archive['species'] == 'mouse']['arrays'].tolist()
        elif self._cohort == 'human_only':
            puck_id_list = df_archive[df_archive['species'] == 'human']['arrays'].tolist()
        else:
            puck_id_list = df_archive['arrays'].tolist()

        all_anndata_files = []
        for file in listdir(self._data_dir):
            if file.startswith("anndata"):
                all_anndata_files.append(file)

        anndata_included, puck_id_included = [], []
        for puck_id in puck_id_list:
            for anndata_file in all_anndata_files:
                if puck_id in anndata_file:
                    anndata_included.append(anndata_file)
                    puck_id_included.append(puck_id)
                    break

        print("len(anndata_included) -->", len(anndata_included))

        # Note that I make some identification and map different cell_types to the same channel
        cell_list = ["CD-IC", "CD-PC", "DCT", "EC", "Fibroblast", "GC", "MC", "vSMC", "Macrophage",
                     "Podocyte", "Other_Immune", "PCT", "PCT_1", "PCT_2", "TAL", "MD"]
        codes_list = [0, 1, 2, 3, 4, 5, 6, 7, 8,
                      9, 10, 11, 11, 11, 12, 12]
        categories_to_codes = dict(zip(cell_list, codes_list))

        all_labels_sparse_images = [array_to_condition[puck_id] for puck_id in puck_id_included]
        all_names_sparse_images = puck_id_included
        all_metadata = [MetadataCropperDataset(f_name=f_name, loc_x=0.0, loc_y=0.0, moran=-99.9) for
                        f_name in all_names_sparse_images]
        all_anndata = [read_h5ad(os.path.join(self._data_dir, anndata)) for anndata in anndata_included]

        # set the count matrix to None (if necessary)
        if not self._load_count_matrix:
            for anndata in all_anndata:
                anndata.X = None

        all_sparse_images = [SparseImage.from_anndata(
            anndata,
            x_key="xcoord",
            y_key="ycoord",
            category_key="cell_type",
            pixel_size=self._pixel_size,
            padding=10,
            categories_to_codes=categories_to_codes) for anndata in all_anndata]

        all_sparse_images_cpu = [sp_image.to(torch.device('cpu')) for sp_image in all_sparse_images]
        torch.save((all_sparse_images_cpu, all_labels_sparse_images, all_metadata),
                   os.path.join(self._data_dir, "train_dataset.pt"))
        print("saved the file", os.path.join(self._data_dir, "train_dataset.pt"))

        # create test_dataset_random and write to file
        list_imgs, list_labels, list_fnames, list_loc_xs, list_loc_ys, list_morans = [], [], [], [], [], []
        for sp_img, label, fname in zip(all_sparse_images, all_labels_sparse_images, all_names_sparse_images):
            sps_tmp, loc_x_tmp, loc_y_tmp = self.cropper_test(sp_img, n_crops=self._n_crops_for_tissue_test)
            list_morans += [self.compute_moran(sparse_tensor).max().item() for sparse_tensor in sps_tmp]
            list_imgs += sps_tmp
            list_labels += [label] * len(sps_tmp)
            list_fnames += [fname] * len(sps_tmp)
            list_loc_xs += loc_x_tmp
            list_loc_ys += loc_y_tmp

        list_metadata = [MetadataCropperDataset(f_name=f_name, loc_x=loc_x, loc_y=loc_y, moran=moran) for
                         f_name, loc_x, loc_y, moran in zip(list_fnames, list_loc_xs, list_loc_ys, list_morans)]
        list_imgs_cpu = [img.cpu() for img in list_imgs]
        torch.save((list_imgs_cpu, list_labels, list_metadata), os.path.join(self._data_dir, "test_dataset.pt"))
        print("saved the file", os.path.join(self._data_dir, "test_dataset.pt"))

    def setup(self, stage: Optional[str] = None) -> None:
        raise NotImplementedError
        # these are things that run on each gpus.
        # Surprisingly, here model.device == cpu while later in dataloader model.device == cuda:0
        """ stage: either 'fit', 'validate', 'test', or 'predict' """
        print("-----> running datamodule setup. stage -> {0}".format(stage))

        # This need to go here so that each gpu has a dictionary available
        df_archive = pd.read_csv(os.path.join(self._data_dir, "slideseq_kidney_datatable.csv"),
                                 usecols=["arrays", "samples", "species"])

        array_list = df_archive["arrays"].tolist()
        species_list = df_archive["species"].tolist()
        condition_list = df_archive["samples"].tolist()

        self.array_to_code = dict(zip(array_list, range(len(array_list))))
        self.array_to_species = dict(zip(array_list, species_list))
        self.array_to_condition = dict(zip(array_list, condition_list))



class DinoSparseFolderDM(DinoSparseDM):
    """
    It expects a folder structure as follow
    root
    ├── train/
    |   ├── 0001.hd5d
    |   └── 0002.h5ad
    └── test/
        ├── 0003.hd5d
        └── 0004.h5ad
    """
    def __init__(self):
        raise NotImplementedError
        # TODO: Implement a general dataset that takes a folder with anndata objects.
    pass
