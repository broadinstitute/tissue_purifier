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
    AddFakeMetadata,
    CropperDataset,
    DataLoaderWithLoad,
    CollateFnListTuple,
    MetadataCropperDataset,
    CropperDenseTensor,
    CropperSparseTensor,
    CropperTensor,
)

# SparseTensor can not be used in dataloader using num_workers > 0. See https://github.com/pytorch/pytorch/issues/20248
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


class DinoDM_from_anndata_folder(DinoDM):
    # TODO: Implement a general dataset that takes a folder with anndata objects.
    pass


class DummyDM(DinoDM):
    def __init__(self):
        super().__init__()
        print("-----> running datamodule init")

        # params for overwriting the abstract property
        self._global_size = 64
        self._local_size = 32
        self._n_global_crops = 2
        self._n_local_crops = 2

        # global params
        self.data_dir = "./"
        self.batch_size_per_gpu = 10
        self.num_workers = 1
        self.gpus = torch.cuda.device_count()

        # extras
        self.load_count_matrix = False

        # specify the transform
        self.global_scale = (0.8, 1.0)
        self.local_scale = (0.5, 0.8)
        self.dropout_range = 0.5
        self.rasterize_sigma = (0.5, 0.2)
        self.occlusion_fraction = (0.1, 0.3)

        # params for all datasets
        self.pixel_size = 1
        self.n_element_min_for_crop = 10

        # Callable on dataset
        self.compute_moran = SpatialAutocorrelation(
            modality='moran',
            n_neighbours=6,
            neigh_correct=False)

        # params for building the test_dataset
        self.n_crops_for_tissue_test = 2
        self.n_crops_for_tissue_train = 2

        # things to save the dataset
        self.dataset_train = None
        self.dataset_test = None

    def get_metadata_to_regress(self, metadata: MetadataCropperDataset) -> Dict[str, float]:
        """ Extract one or more quantities to regress from the metadata """
        return {
            "moran": float(metadata.moran),
            "loc_x": float(metadata.loc_x),
        }

    def get_metadata_to_classify(self, metadata: MetadataCropperDataset) -> Dict[str, int]:
        """ Extract one or more quantities to classify from the metadata """
        conversion1 = {
            'id0': 0,
            'id1': 1,
        }

        return {
            "condition": conversion1[metadata.f_name],
        }

    @classproperty
    def ch_in(self) -> int:
        return 9

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
            n_element_min=self.n_element_min_for_crop,
            n_crops=self.n_crops_for_tissue_test,
            random_order=True,
        )

    @property
    def cropper_train(self):
        return CropperSparseTensor(
            strategy='random',
            crop_size=int(self._global_size * 1.5),
            n_element_min=int(self.n_element_min_for_crop * 1.5 * 1.5),
            n_crops=self.n_crops_for_tissue_train,
            random_order=True,
        )

    @property
    def trsfm_test(self) -> Callable:
        return TransformForList(
            transform_before_stack=torchvision.transforms.Compose([
                DropoutSparseTensor(p=0.5, dropout_rate=self.dropout_range),
                SparseToDense(),
                Rasterize(sigma=self.rasterize_sigma, normalize=False),
                RandomVFlip(p=0.5),
                RandomHFlip(p=0.5),
                RandomGlobalIntensity(f_min=0.9, f_max=1.1)
            ]),
            transform_after_stack=torchvision.transforms.CenterCrop(size=self.global_size),
        )

    @property
    def trsfm_train_global(self) -> Callable:
        return TransformForList(
            transform_before_stack=torchvision.transforms.Compose([
                DropoutSparseTensor(p=0.5, dropout_rate=self.dropout_range),
                SparseToDense(),
                RandomGlobalIntensity(f_min=0.9, f_max=1.1)
            ]),
            transform_after_stack=torchvision.transforms.Compose([
                Rasterize(sigma=self.rasterize_sigma, normalize=False),
                torchvision.transforms.RandomRotation(
                    degrees=(-180.0, 180.0),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    expand=False,
                    fill=0.0),
                torchvision.transforms.CenterCrop(size=self.global_size),
                RandomVFlip(p=0.5),
                RandomHFlip(p=0.5),
                torchvision.transforms.RandomResizedCrop(
                    size=(self.global_size, self.global_size),
                    scale=self.global_scale,
                    ratio=(0.95, 1.05),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                RandomStraightCut(p=0.5, occlusion_fraction=self.occlusion_fraction),
            ])
        )

    @property
    def trsfm_train_local(self) -> Callable:
        return TransformForList(
            transform_before_stack=torchvision.transforms.Compose([
                DropoutSparseTensor(p=0.5, dropout_rate=self.dropout_range),
                SparseToDense(),
                RandomGlobalIntensity(f_min=0.9, f_max=1.1)
            ]),
            transform_after_stack=torchvision.transforms.Compose([
                Rasterize(sigma=self.rasterize_sigma, normalize=False),
                torchvision.transforms.RandomRotation(
                    degrees=(-180.0, 180.0),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    expand=False,
                    fill=0.0),
                torchvision.transforms.CenterCrop(size=self.global_size),
                RandomVFlip(p=0.5),
                RandomHFlip(p=0.5),
                torchvision.transforms.RandomResizedCrop(
                    size=(self.local_size, self.local_size),
                    scale=self.local_scale,
                    ratio=(0.95, 1.05),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                RandomStraightCut(p=0.5, occlusion_fraction=self.occlusion_fraction),
            ])
        )

    def prepare_data(self):
        # these are things to be done only once in distributed settings
        # good for writing stuff to disk and avoid corruption
        print("-----> running datamodule prepare_data")

        cell_list = ["ES", "Endothelial", "Leydig", "Macrophage", "Myoid", "RS", "SPC", "SPG", "Sertoli"]
        categories_to_codes = dict(zip(cell_list, range(len(cell_list))))

        import random
        import pandas
        import numpy
        from anndata import AnnData

        n_tissues, n_beads = 2, 1000
        all_anndata, all_names_sparse_images, all_labels_sparse_images = [], [], []
        for n_tissue in range(n_tissues):
            tmp_dict = {
                "x_raw": 200.0 + 100.0 * numpy.random.rand(n_beads),
                "y_raw": 200.0 + 100.0 * numpy.random.rand(n_beads),
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
            pixel_size=self.pixel_size,
            padding=10,
            categories_to_codes=categories_to_codes) for anndata in all_anndata]

        all_sparse_images_cpu = [sp_image.cpu() for sp_image in all_sparse_images]
        torch.save((all_sparse_images_cpu, all_labels_sparse_images, all_metadata),
                   os.path.join(self.data_dir, "train_dataset.pt"))
        print("saved the file", os.path.join(self.data_dir, "train_dataset.pt"))

        # create test_dataset_random and write to file
        list_imgs, list_labels, list_fnames, list_loc_xs, list_loc_ys, list_morans = [], [], [], [], [], []
        for sp_img, label, fname in zip(all_sparse_images, all_labels_sparse_images, all_names_sparse_images):
            sps_tmp, loc_x_tmp, loc_y_tmp = self.cropper_test(sp_img, n_crops=self.n_crops_for_tissue_test)
            list_morans += [self.compute_moran(sparse_tensor).max().item() for sparse_tensor in sps_tmp]
            list_imgs += sps_tmp
            list_labels += [label] * len(sps_tmp)
            list_fnames += [fname] * len(sps_tmp)
            list_loc_xs += loc_x_tmp
            list_loc_ys += loc_y_tmp

        list_metadata = [MetadataCropperDataset(f_name=f_name, loc_x=loc_x, loc_y=loc_y, moran=moran) for
                         f_name, loc_x, loc_y, moran in zip(list_fnames, list_loc_xs, list_loc_ys, list_morans)]
        list_imgs_cpu = [img.cpu() for img in list_imgs]
        torch.save((list_imgs_cpu, list_labels, list_metadata), os.path.join(self.data_dir, "test_dataset.pt"))
        print("saved the file", os.path.join(self.data_dir, "test_dataset.pt"))

    def setup(self, stage: Optional[str] = None) -> None:
        # these are things that run on each gpus.
        # Surprisingly, here self.trainer.model.device == cpu
        # while later in dataloader self.trainer.model.device == cuda:0
        """ stage: either 'fit', 'validate', 'test', or 'predict' """
        print("-----> running datamodule setup. stage -> {0}".format(stage))

        list_imgs, list_labels, list_metadata = torch.load(os.path.join(self.data_dir, "train_dataset.pt"))
        list_imgs = [img.coalesce().cpu() for img in list_imgs]
        self.dataset_train = CropperDataset(
            imgs=list_imgs,
            labels=list_labels,
            metadatas=list_metadata,
            cropper=self.cropper_train,
        )
        print("created train_dataset device = {0}, length = {1}".format(self.dataset_train.imgs[0].device,
                                                                        self.dataset_train.__len__()))

        list_imgs, list_labels, list_metadata = torch.load(os.path.join(self.data_dir, "test_dataset.pt"))
        list_imgs = [img.coalesce().cpu() for img in list_imgs]
        self.dataset_test = CropperDataset(
            imgs=list_imgs,
            labels=list_labels,
            metadatas=list_metadata,
            cropper=None,
        )
        print("created test_dataset device = {0}, length = {1}".format(self.dataset_test.imgs[0].device,
                                                                       self.dataset_test.__len__()))

    def train_dataloader(self) -> DataLoaderWithLoad:
        try:
            device = self.trainer.model.device
        except AttributeError:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Inside train_dataloader", device)

        dataloader_train = DataLoaderWithLoad(
            dataset=self.dataset_train.to(device),
            batch_size=self.batch_size_per_gpu,
            collate_fn=CollateFnListTuple(),
            num_workers=0,  # problem if this is larger than 0, see https://github.com/pytorch/pytorch/issues/20248
            shuffle=True,
            drop_last=True,
        )
        return dataloader_train

    def val_dataloader(self) -> List[DataLoaderWithLoad]:  # the same as test
        try:
            device = self.trainer.model.device
        except AttributeError:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Inside val_dataloader", device)

        test_dataloader = DataLoaderWithLoad(
            dataset=self.dataset_test.to(device),
            batch_size=self.batch_size_per_gpu,
            collate_fn=CollateFnListTuple(),
            num_workers=0,  # problem if this is larger than 0, see https://github.com/pytorch/pytorch/issues/20248
            shuffle=False,
            drop_last=False,
        )
        return [test_dataloader]


class SlideSeqTestisDM(DinoDM):
    def __init__(
            self,
            # stuff to redefine abstract properties
            global_size: int,
            local_size: int,
            n_local_crops: int,
            n_global_crops: int,
            # extrax
            load_count_matrix: bool,

            # additional arguments
            num_workers: int,
            data_dir: str,
            batch_size_per_gpu: int,
            n_neighbours_moran: int,
            pixel_size: int,
            # specify the transform
            global_scale: Tuple[float, float],
            local_scale: Tuple[float, float],
            n_element_min_for_crop: int,
            dropout_range: Tuple[float, float],
            rasterize_sigma: Tuple[float, float],
            occlusion_fraction: Tuple[float, float],
            n_crops_for_tissue_test: int,
            n_crops_for_tissue_train: int,
            **kargs,
    ):
        super().__init__()
        print("-----> running datamodule init")

        # params for overwriting the abstract property
        self._global_size = global_size
        self._local_size = local_size
        self._n_global_crops = n_global_crops
        self._n_local_crops = n_local_crops

        # global params
        self.data_dir = data_dir
        self.batch_size_per_gpu = batch_size_per_gpu
        self.num_workers = cpu_count() if num_workers == -1 else num_workers
        self.gpus = torch.cuda.device_count()

        # extras
        self.load_count_matrix = load_count_matrix

        # specify the transform
        self.global_scale = global_scale
        self.local_scale = local_scale
        self.dropout_range = dropout_range
        self.rasterize_sigma = rasterize_sigma
        self.occlusion_fraction = occlusion_fraction

        # params for all datasets
        self.pixel_size = pixel_size
        self.n_element_min_for_crop = n_element_min_for_crop

        # Callable on dataset
        self.compute_moran = SpatialAutocorrelation(
            modality='moran',
            n_neighbours=n_neighbours_moran,
            neigh_correct=False)

        # params for building the test_dataset
        self.n_crops_for_tissue_test = n_crops_for_tissue_test
        self.n_crops_for_tissue_train = n_crops_for_tissue_train

        # things to save the dataset
        self.dataset_train = None
        self.dataset_test = None

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
            n_element_min=self.n_element_min_for_crop,
            n_crops=self.n_crops_for_tissue_test,
            random_order=True,
        )

    @property
    def cropper_train(self):
        return CropperSparseTensor(
            strategy='random',
            crop_size=int(self._global_size * 1.5),
            n_element_min=int(self.n_element_min_for_crop * 1.5 * 1.5),
            n_crops=self.n_crops_for_tissue_train,
            random_order=True,
        )

    @property
    def trsfm_test(self) -> Callable:
        return TransformForList(
            transform_before_stack=torchvision.transforms.Compose([
                DropoutSparseTensor(p=0.5, dropout_rate=self.dropout_range),
                SparseToDense(),
                Rasterize(sigma=self.rasterize_sigma, normalize=False),
                RandomVFlip(p=0.5),
                RandomHFlip(p=0.5),
                RandomGlobalIntensity(f_min=0.9, f_max=1.1)
            ]),
            transform_after_stack=torchvision.transforms.CenterCrop(size=self.global_size),
        )

    @property
    def trsfm_train_global(self) -> Callable:
        return TransformForList(
            transform_before_stack=torchvision.transforms.Compose([
                DropoutSparseTensor(p=0.5, dropout_rate=self.dropout_range),
                SparseToDense(),
                RandomGlobalIntensity(f_min=0.9, f_max=1.1)
            ]),
            transform_after_stack=torchvision.transforms.Compose([
                Rasterize(sigma=self.rasterize_sigma, normalize=False),
                torchvision.transforms.RandomRotation(
                    degrees=(-180.0, 180.0),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    expand=False,
                    fill=0.0),
                torchvision.transforms.CenterCrop(size=self.global_size),
                RandomVFlip(p=0.5),
                RandomHFlip(p=0.5),
                torchvision.transforms.RandomResizedCrop(
                    size=(self.global_size, self.global_size),
                    scale=self.global_scale,
                    ratio=(0.95, 1.05),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                RandomStraightCut(p=0.5, occlusion_fraction=self.occlusion_fraction),
            ])
        )

    @property
    def trsfm_train_local(self) -> Callable:
        return TransformForList(
            transform_before_stack=torchvision.transforms.Compose([
                DropoutSparseTensor(p=0.5, dropout_rate=self.dropout_range),
                SparseToDense(),
                RandomGlobalIntensity(f_min=0.9, f_max=1.1)
            ]),
            transform_after_stack=torchvision.transforms.Compose([
                Rasterize(sigma=self.rasterize_sigma, normalize=False),
                torchvision.transforms.RandomRotation(
                    degrees=(-180.0, 180.0),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    expand=False,
                    fill=0.0),
                torchvision.transforms.CenterCrop(size=self.global_size),
                RandomVFlip(p=0.5),
                RandomHFlip(p=0.5),
                torchvision.transforms.RandomResizedCrop(
                    size=(self.local_size, self.local_size),
                    scale=self.local_scale,
                    ratio=(0.95, 1.05),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                RandomStraightCut(p=0.5, occlusion_fraction=self.occlusion_fraction),
            ])
        )

    @classmethod
    def add_specific_args(cls, parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')

        # global parameters
        parser.add_argument("--data_dir", type=str, default="./slide_seq_testis",
                            help="directory where to download the data")
        parser.add_argument("--num_workers", default=cpu_count(), type=int,
                            help="number of worker to load data. Meaningful only if dataset is on disk. \
                            Set to zero if data in memory")

        parser.add_argument("--n_neighbours_moran", type=int, default=6,
                            help="number of neighbours used to compute moran")
        parser.add_argument("--pixel_size", type=float, default=4.0,
                            help="size of the pixel to convert csv to sparse image")

        # extras
        parser.add_argument("--load_count_matrix", type=smart_bool, default=False,
                            help="If true load the count matrix in the anndata object. \
                                  Count matrix is memory intensive therefore it can be advantegeous not to load it.")

        # specify the transform
        parser.add_argument("--n_element_min_for_crop", type=int, default=200,
                            help="minimum number of beads/cell in a crop")
        parser.add_argument("--dropout_range", type=float, nargs=2, default=[0.1, 0.3],
                            help="Dropout range should be in (0.0,1.0)")
        parser.add_argument("--rasterize_sigma", type=float, nargs=2, default=[1.0, 2.0],
                            help="Sigma of the gaussian kernel used for rasterization")
        parser.add_argument("--occlusion_fraction", type=float, nargs=2, default=[0.1, 0.3],
                            help="Fraction of the sample which might be occluded. Should be in (0.0, 1.0)")
        parser.add_argument("--global_size", type=int, default=128, help="size in pixel of the global crops")
        parser.add_argument("--local_size", type=int, default=64, help="size in pixel of the local crops")
        parser.add_argument("--global_scale", type=float, nargs=2, default=[0.8, 1.0],
                            help="scale used in RandomResizedCrop for the global crops")
        parser.add_argument("--local_scale", type=float, nargs=2, default=[0.4, 0.5],
                            help="scale used in RandomResizedCrop for the local crops")
        parser.add_argument("--n_local_crops", type=int, default=6, help="number of local crops")
        parser.add_argument("--n_global_crops", type=int, default=2, help="number of global crops")

        # test dataset
        parser.add_argument("--n_crops_for_tissue_train", type=int, default=600,
                            help="The number of crops in each training epoch will be: n_tissue * n_crops. \
                            Set small for rapid prototyping")
        parser.add_argument("--n_crops_for_tissue_test", type=int, default=600,
                            help="The number of crops in each test epoch will be: n_tissue * n_crops. \
                            Set small for rapid prototyping")
        parser.add_argument("--batch_size_per_gpu", type=int, default=64,
                            help="Batch size FOR EACH GPUs. Set small for rapid prototyping. \
                            The total batch_size will increase linearly with the number of GPUs. \
                            If strategy == 'tiling' then the actual batch_size might be smaller than this one")

        return parser

    def __tar_exists__(self):
        return os.path.exists(os.path.join(self.data_dir, "slideseq_testis_anndata_h5ad.tar.gz"))

    def __download__tar__(self):
        from google.cloud import storage

        bucket_name = "ld-data-bucket"
        source_blob_name = "tissue-purifier/slideseq_testis_anndata_h5ad.tar.gz"
        destination_file_name = os.path.join(self.data_dir, "slideseq_testis_anndata_h5ad.tar.gz")

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

    def __untar__(self):
        tar_file_name = os.path.join(self.data_dir, "slideseq_testis_anndata_h5ad.tar.gz")

        # untar the tar.gz file
        with tarfile.open(tar_file_name, "r:gz") as fp:
            fp.extractall(path=self.data_dir)

    def prepare_data(self):
        # these are things to be done only once in distributed settings
        # good for writing stuff to disk and avoid corruption
        print("-----> running datamodule prepare_data")

        print("Will create the test and train file")
        if self.__tar_exists__():
            print("untar data")
            self.__untar__()
        else:
            print("download and untar data")
            self.__download__tar__()
            self.__untar__()

        anndata_wt1 = read_h5ad(os.path.join(self.data_dir, "anndata_wt1.h5ad"))
        anndata_wt2 = read_h5ad(os.path.join(self.data_dir, "anndata_wt2.h5ad"))
        anndata_wt3 = read_h5ad(os.path.join(self.data_dir, "anndata_wt3.h5ad"))
        anndata_sick1 = read_h5ad(os.path.join(self.data_dir, "anndata_sick1.h5ad"))
        anndata_sick2 = read_h5ad(os.path.join(self.data_dir, "anndata_sick2.h5ad"))
        anndata_sick3 = read_h5ad(os.path.join(self.data_dir, "anndata_sick3.h5ad"))

        # create train_dataset and write to file
        cell_list = ["ES", "Endothelial", "Leydig", "Macrophage", "Myoid", "RS", "SPC", "SPG", "Sertoli"]
        categories_to_codes = dict(zip(cell_list, range(len(cell_list))))
        all_anndata = [anndata_wt1, anndata_wt2, anndata_wt3, anndata_sick1, anndata_sick2, anndata_sick3]
        all_labels_sparse_images = ['wt', 'wt', 'wt', 'sick', 'sick', 'sick']
        all_names_sparse_images = ["wt1", "wt2", "wt3", "dis1", "dis2", "dis3"]
        all_metadata = [MetadataCropperDataset(f_name=f_name, loc_x=0.0, loc_y=0.0, moran=-99.9) for
                        f_name in all_names_sparse_images]

        # set the count matrix to None (if necessary)
        if not self.load_count_matrix:
            for anndata in all_anndata:
                anndata.X = None

        # create the train_dataset and write to file
        all_sparse_images = [SparseImage.from_anndata(
            anndata,
            x_key="x",
            y_key="y",
            category_key="cell_type",
            pixel_size=self.pixel_size,
            padding=10,
            categories_to_codes=categories_to_codes) for anndata in all_anndata]

        all_sparse_images_cpu = [sp_image.cpu() for sp_image in all_sparse_images]
        torch.save((all_sparse_images_cpu, all_labels_sparse_images, all_metadata),
                   os.path.join(self.data_dir, "train_dataset.pt"))
        print("saved the file", os.path.join(self.data_dir, "train_dataset.pt"))

        # create test_dataset_random and write to file
        list_imgs, list_labels, list_fnames, list_loc_xs, list_loc_ys, list_morans = [], [], [], [], [], []
        for sp_img, label, fname in zip(all_sparse_images, all_labels_sparse_images, all_names_sparse_images):
            sps_tmp, loc_x_tmp, loc_y_tmp = self.cropper_test(sp_img, n_crops=self.n_crops_for_tissue_test)
            list_morans += [self.compute_moran(sparse_tensor).max().item() for sparse_tensor in sps_tmp]
            list_imgs += sps_tmp
            list_labels += [label] * len(sps_tmp)
            list_fnames += [fname] * len(sps_tmp)
            list_loc_xs += loc_x_tmp
            list_loc_ys += loc_y_tmp

        list_metadata = [MetadataCropperDataset(f_name=f_name, loc_x=loc_x, loc_y=loc_y, moran=moran) for
                         f_name, loc_x, loc_y, moran in zip(list_fnames, list_loc_xs, list_loc_ys, list_morans)]
        list_imgs_cpu = [img.cpu() for img in list_imgs]
        torch.save((list_imgs_cpu, list_labels, list_metadata), os.path.join(self.data_dir, "test_dataset.pt"))
        print("saved the file", os.path.join(self.data_dir, "test_dataset.pt"))

    def setup(self, stage: Optional[str] = None) -> None:
        # these are things that run on each gpus.
        # Surprisingly, here self.trainer.model.device == cpu
        # while later in dataloader self.trainer.model.device == cuda:0
        """ stage: either 'fit', 'validate', 'test', or 'predict' """
        print("-----> running datamodule setup. stage -> {0}".format(stage))

        list_imgs, list_labels, list_metadata = torch.load(os.path.join(self.data_dir, "train_dataset.pt"))
        list_imgs = [img.coalesce().cpu() for img in list_imgs]
        self.dataset_train = CropperDataset(
            imgs=list_imgs,
            labels=list_labels,
            metadatas=list_metadata,
            cropper=self.cropper_train,
        )
        print("created train_dataset device = {0}, length = {1}".format(self.dataset_train.imgs[0].device,
                                                                        self.dataset_train.__len__()))

        list_imgs, list_labels, list_metadata = torch.load(os.path.join(self.data_dir, "test_dataset.pt"))
        list_imgs = [img.coalesce().cpu() for img in list_imgs]
        self.dataset_test = CropperDataset(
            imgs=list_imgs,
            labels=list_labels,
            metadatas=list_metadata,
            cropper=None,
        )
        print("created test_dataset device = {0}, length = {1}".format(self.dataset_test.imgs[0].device,
                                                                       self.dataset_test.__len__()))

    def train_dataloader(self) -> DataLoaderWithLoad:
        try:
            device = self.trainer.model.device
        except AttributeError:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Inside train_dataloader", device)

        dataloader_train = DataLoaderWithLoad(
            dataset=self.dataset_train.to(device),
            batch_size=self.batch_size_per_gpu,
            collate_fn=CollateFnListTuple(),
            num_workers=0,  # problem if this is larger than 0, see https://github.com/pytorch/pytorch/issues/20248
            shuffle=True,
            drop_last=True,
        )
        return dataloader_train

    def val_dataloader(self) -> List[DataLoaderWithLoad]:  # the same as test
        try:
            device = self.trainer.model.device
        except AttributeError:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Inside val_dataloader", device)

        test_dataloader = DataLoaderWithLoad(
            dataset=self.dataset_test.to(device),
            batch_size=self.batch_size_per_gpu,
            collate_fn=CollateFnListTuple(),
            num_workers=0,  # problem if this is larger than 0, see https://github.com/pytorch/pytorch/issues/20248
            shuffle=False,
            drop_last=False,
        )
        return [test_dataloader]


class SlideSeqKidneyDM(DinoDM):
    # TODO: do the dataloaders as in Testis dataset

    def __init__(
            self,
            cohort: str,
            # stuff to redefine abstract properties
            global_size: int,
            local_size: int,
            n_local_crops: int,
            n_global_crops: int,
            # extras
            load_count_matrix: bool,

            # additional arguments
            num_workers: int,
            data_dir: str,
            batch_size_per_gpu: int,
            n_neighbours_moran: int,
            pixel_size: int,
            # specify the transform
            global_scale: Tuple[float, float],
            local_scale: Tuple[float, float],
            n_element_min_for_crop: int,
            dropout_range: Tuple[float, float],
            rasterize_sigma: Tuple[float, float],
            occlusion_fraction: Tuple[float, float],
            n_crops_for_tissue_test: int,
            n_crops_for_tissue_train: int,
            **kargs,
    ):
        super().__init__()
        print("-----> running datamodule init")

        self.cohort = cohort

        # params for overwriting the abstract property
        self._global_size = global_size
        self._local_size = local_size
        self._n_global_crops = n_global_crops
        self._n_local_crops = n_local_crops

        # global params
        self.data_dir = data_dir
        self.batch_size_per_gpu = batch_size_per_gpu
        self.num_workers = cpu_count() if num_workers == -1 else num_workers
        self.gpus = torch.cuda.device_count()

        # extras
        self.load_count_matrix = load_count_matrix

        # specify the transform
        self.global_scale = global_scale
        self.local_scale = local_scale
        self.dropout_range = dropout_range
        self.rasterize_sigma = rasterize_sigma
        self.occlusion_fraction = occlusion_fraction

        # params for all datasets
        self.pixel_size = pixel_size
        self.n_element_min_for_crop = n_element_min_for_crop

        # Callable on dataset
        self.compute_moran = SpatialAutocorrelation(
            modality='moran',
            n_neighbours=n_neighbours_moran,
            neigh_correct=False)

        # params for building the test_dataset
        self.n_crops_for_tissue_test = n_crops_for_tissue_test
        self.n_crops_for_tissue_train = n_crops_for_tissue_train

        # dictionary which will be created during prepare_data
        self.array_to_code = None
        self.array_to_species = None
        self.array_to_condition = None

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
            n_element_min=self.n_element_min_for_crop,
            n_crops=self.n_crops_for_tissue_test,
            random_order=True,
        )

    @property
    def cropper_train(self):
        return CropperSparseTensor(
            strategy='random',
            crop_size=int(self._global_size * 1.5),
            n_element_min=int(self.n_element_min_for_crop * 1.5 * 1.5),
            n_crops=self.n_crops_for_tissue_train,
            random_order=True,
        )

    @property
    def trsfm_test(self) -> Callable:
        return TransformForList(
            transform_before_stack=torchvision.transforms.Compose([
                DropoutSparseTensor(p=0.5, dropout_rate=self.dropout_range),
                SparseToDense(),
                Rasterize(sigma=self.rasterize_sigma, normalize=False),
                RandomVFlip(p=0.5),
                RandomHFlip(p=0.5),
                RandomGlobalIntensity(f_min=0.9, f_max=1.1),
            ]),
            transform_after_stack=torchvision.transforms.CenterCrop(size=self.global_size),
        )

    @property
    def trsfm_train_global(self) -> Callable:
        return TransformForList(
            transform_before_stack=torchvision.transforms.Compose([
                DropoutSparseTensor(p=0.5, dropout_rate=self.dropout_range),
                SparseToDense(),
                RandomGlobalIntensity(f_min=0.9, f_max=1.1)
            ]),
            transform_after_stack=torchvision.transforms.Compose([
                Rasterize(sigma=self.rasterize_sigma, normalize=False),
                torchvision.transforms.RandomRotation(
                    degrees=(-180.0, 180.0),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    expand=False,
                    fill=0.0),
                torchvision.transforms.CenterCrop(size=self.global_size),
                RandomVFlip(p=0.5),
                RandomHFlip(p=0.5),
                torchvision.transforms.RandomResizedCrop(
                    size=(self.global_size, self.global_size),
                    scale=self.global_scale,
                    ratio=(0.95, 1.05),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                RandomStraightCut(p=0.5, occlusion_fraction=self.occlusion_fraction),
            ])
        )

    @property
    def trsfm_train_local(self) -> Callable:
        return TransformForList(
            transform_before_stack=torchvision.transforms.Compose([
                DropoutSparseTensor(p=0.5, dropout_rate=self.dropout_range),
                SparseToDense(),
                RandomGlobalIntensity(f_min=0.9, f_max=1.1)
            ]),
            transform_after_stack=torchvision.transforms.Compose([
                Rasterize(sigma=self.rasterize_sigma, normalize=False),
                torchvision.transforms.RandomRotation(
                    degrees=(-180.0, 180.0),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    expand=False,
                    fill=0.0),
                torchvision.transforms.CenterCrop(size=self.global_size),
                RandomVFlip(p=0.5),
                RandomHFlip(p=0.5),
                torchvision.transforms.RandomResizedCrop(
                    size=(self.local_size, self.local_size),
                    scale=self.local_scale,
                    ratio=(0.95, 1.05),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                RandomStraightCut(p=0.5, occlusion_fraction=self.occlusion_fraction),
            ])
        )

    @classmethod
    def add_specific_args(cls, parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')

        # global parameters
        parser.add_argument("--cohort", default='test', type=str,
                            choices=['test', 'all', 'mouse_only', 'human_only'],
                            help="Specify grouping of samples to use during training and testing")
        parser.add_argument("--data_dir", type=str, default="./slide_seq_kidney",
                            help="directory where to download the data")
        parser.add_argument("--num_workers", default=cpu_count(), type=int,
                            help="number of worker to load data. Meaningful only if dataset is on disk. \
                            Set to zero if data in memory")

        parser.add_argument("--n_neighbours_moran", type=int, default=6,
                            help="number of neighbours used to compute moran")
        parser.add_argument("--pixel_size", type=float, default=4.0,
                            help="size of the pixel to convert csv to sparse image")

        # extras
        parser.add_argument("--load_count_matrix", type=smart_bool, default=False,
                            help="If true load the count matrix in the anndata object. \
                                  Count matrix is memory intensive therefore it can be advantegeous not to load it.")

        # specify the transform
        parser.add_argument("--n_element_min_for_crop", type=int, default=200,
                            help="minimum number of beads/cell in a crop")
        parser.add_argument("--dropout_range", type=float, nargs=2, default=[0.1, 0.3],
                            help="Dropout range should be in (0.0,1.0)")
        parser.add_argument("--rasterize_sigma", type=float, nargs=2, default=[1.0, 2.0],
                            help="Sigma of the gaussian kernel used for rasterization")
        parser.add_argument("--occlusion_fraction", type=float, nargs=2, default=[0.1, 0.3],
                            help="Fraction of the sample which might be occluded. Should be in (0.0, 1.0)")
        parser.add_argument("--global_size", type=int, default=128, help="size in pixel of the global crops")
        parser.add_argument("--local_size", type=int, default=64, help="size in pixel of the local crops")
        parser.add_argument("--global_scale", type=float, nargs=2, default=[0.8, 1.0],
                            help="scale used in RandomResizedCrop for the global crops")
        parser.add_argument("--local_scale", type=float, nargs=2, default=[0.4, 0.5],
                            help="scale used in RandomResizedCrop for the local crops")
        parser.add_argument("--n_local_crops", type=int, default=6, help="number of local crops")
        parser.add_argument("--n_global_crops", type=int, default=2, help="number of global crops")

        # test dataset
        parser.add_argument("--n_crops_for_tissue_train", type=int, default=30,
                            help="The number of crops in each training epoch will be: n_tissue * n_crops. \
                            Set small for rapid prototyping")
        parser.add_argument("--n_crops_for_tissue_test", type=int, default=30,
                            help="The number of crops in each test epoch will be: n_tissue * n_crops. \
                            Set small for rapid prototyping")
        parser.add_argument("--batch_size_per_gpu", type=int, default=64,
                            help="Batch size FOR EACH GPUs. Set small for rapid prototyping. \
                            The total batch_size will increase linearly with the number of GPUs. \
                            If strategy == 'tiling' then the actual batch_size might be smaller than this one")

        return parser

    def __tar_exists__(self):
        return os.path.exists(os.path.join(self.data_dir, "slideseq_kidney_anndata_h5ad.tar.gz"))

    def __download__tar__(self):
        from google.cloud import storage

        bucket_name = "ld-data-bucket"
        source_blob_name = "tissue-purifier/slideseq_kidney_anndata_h5ad.tar.gz"
        destination_file_name = os.path.join(self.data_dir, "slideseq_kidney_anndata_h5ad.tar.gz")

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

    def __untar__(self):
        tar_file_name = os.path.join(self.data_dir, "slideseq_kidney_anndata_h5ad.tar.gz")

        # untar the tar.gz file
        with tarfile.open(tar_file_name, "r:gz") as fp:
            fp.extractall(path=self.data_dir)

    def prepare_data(self):
        # these are things to be done only once in distributed settings
        print("-----> running datamodule prepare_data")

        print("Will create the test and train file")
        if self.__tar_exists__():
            print("untar data")
            self.__untar__()
        else:
            print("download and untar data")
            self.__download__tar__()
            self.__untar__()

        df_archive = pd.read_csv(os.path.join(self.data_dir, "slideseq_kidney_datatable.csv"),
                                 usecols=["arrays", "samples", "species"])
        array_list = df_archive["arrays"].tolist()
        condition_list = df_archive["samples"].tolist()
        array_to_condition = dict(zip(array_list, condition_list))

        print("COHORT -->", self.cohort)
        if self.cohort == 'test':
            puck_id_list = df_archive.head(6)['arrays'].tolist()
        elif self.cohort == 'mouse_only':
            puck_id_list = df_archive[df_archive['species'] == 'mouse']['arrays'].tolist()
        elif self.cohort == 'human_only':
            puck_id_list = df_archive[df_archive['species'] == 'human']['arrays'].tolist()
        else:
            puck_id_list = df_archive['arrays'].tolist()

        all_anndata_files = []
        for file in listdir(self.data_dir):
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
        all_anndata = [read_h5ad(os.path.join(self.data_dir, anndata)) for anndata in anndata_included]

        # set the count matrix to None (if necessary)
        if not self.load_count_matrix:
            for anndata in all_anndata:
                anndata.X = None

        all_sparse_images = [SparseImage.from_anndata(
            anndata,
            x_key="xcoord",
            y_key="ycoord",
            category_key="cell_type",
            pixel_size=self.pixel_size,
            padding=10,
            categories_to_codes=categories_to_codes) for anndata in all_anndata]

        all_sparse_images_cpu = [sp_image.to(torch.device('cpu')) for sp_image in all_sparse_images]
        torch.save((all_sparse_images_cpu, all_labels_sparse_images, all_metadata),
                   os.path.join(self.data_dir, "train_dataset.pt"))
        print("saved the file", os.path.join(self.data_dir, "train_dataset.pt"))

        # create test_dataset_random and write to file
        list_imgs, list_labels, list_fnames, list_loc_xs, list_loc_ys, list_morans = [], [], [], [], [], []
        for sp_img, label, fname in zip(all_sparse_images, all_labels_sparse_images, all_names_sparse_images):
            sps_tmp, loc_x_tmp, loc_y_tmp = self.cropper_test(sp_img, n_crops=self.n_crops_for_tissue_test)
            list_morans += [self.compute_moran(sparse_tensor).max().item() for sparse_tensor in sps_tmp]
            list_imgs += sps_tmp
            list_labels += [label] * len(sps_tmp)
            list_fnames += [fname] * len(sps_tmp)
            list_loc_xs += loc_x_tmp
            list_loc_ys += loc_y_tmp

        list_metadata = [MetadataCropperDataset(f_name=f_name, loc_x=loc_x, loc_y=loc_y, moran=moran) for
                         f_name, loc_x, loc_y, moran in zip(list_fnames, list_loc_xs, list_loc_ys, list_morans)]
        list_imgs_cpu = [img.cpu() for img in list_imgs]
        torch.save((list_imgs_cpu, list_labels, list_metadata), os.path.join(self.data_dir, "test_dataset.pt"))
        print("saved the file", os.path.join(self.data_dir, "test_dataset.pt"))

    def setup(self, stage: Optional[str] = None) -> None:
        # these are things that run on each gpus.
        # Surprisingly, here model.device == cpu while later in dataloader model.device == cuda:0
        """ stage: either 'fit', 'validate', 'test', or 'predict' """
        print("-----> running datamodule setup. stage -> {0}".format(stage))

        # This need to go here so that each gpu has a dictionary available
        df_archive = pd.read_csv(os.path.join(self.data_dir, "slideseq_kidney_datatable.csv"),
                                 usecols=["arrays", "samples", "species"])

        array_list = df_archive["arrays"].tolist()
        species_list = df_archive["species"].tolist()
        condition_list = df_archive["samples"].tolist()

        self.array_to_code = dict(zip(array_list, range(len(array_list))))
        self.array_to_species = dict(zip(array_list, species_list))
        self.array_to_condition = dict(zip(array_list, condition_list))

    def train_dataloader(self) -> DataLoaderWithLoad:
        try:
            device = self.trainer.model.device
        except AttributeError:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Inside train_dataloader", device)

        # Read the imgs from disk and put them on the right CUDA memory
        list_imgs_cpu, list_labels, list_metadata = torch.load(os.path.join(self.data_dir, "train_dataset.pt"))
        list_imgs = [img.coalesce().to(device) for img in list_imgs_cpu]

        dataset_train = CropperDataset(
            imgs=list_imgs,
            labels=list_labels,
            metadatas=list_metadata,
            cropper=self.cropper_train,
        )
        print("created train_dataset device = {0}, length = {1}".format(dataset_train.imgs[0].device,
                                                                        dataset_train.__len__()))

        dataloader_train = DataLoaderWithLoad(
            dataset=dataset_train,
            batch_size=self.batch_size_per_gpu,
            collate_fn=CollateFnListTuple(),
            num_workers=0,  # problem if this is larger than 0, see https://github.com/pytorch/pytorch/issues/20248
            shuffle=True,
            drop_last=True,
        )
        return dataloader_train

    def val_dataloader(self) -> List[DataLoaderWithLoad]:  # the same as test
        try:
            device = self.trainer.model.device
        except AttributeError:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Inside test_dataloader", device)

        list_imgs, list_labels, list_metadata = torch.load(os.path.join(self.data_dir, "test_dataset.pt"))
        list_imgs = [img.coalesce().to(device) for img in list_imgs]
        dataset_test = CropperDataset(
            imgs=list_imgs,
            labels=list_labels,
            metadatas=list_metadata,
            cropper=None,
        )
        print("created test_dataset device = {0}, length = {1}".format(dataset_test.imgs[0].device,
                                                                       dataset_test.__len__()))

        test_dataloader = DataLoaderWithLoad(
            dataset=dataset_test,
            batch_size=self.batch_size_per_gpu,
            collate_fn=CollateFnListTuple(),
            num_workers=0,  # problem if this is larger than 0, see https://github.com/pytorch/pytorch/issues/20248
            shuffle=False,
            drop_last=False,
        )
        return [test_dataloader]


#####class CaltechDM(DinoDM):
#####    def __init__(
#####            self,
#####            # stuff to redefine abstract properties
#####            global_size: int,
#####            local_size: int,
#####            n_local_crops: int,
#####            n_global_crops: int,
#####
#####            # additional arguments
#####            num_workers: int,
#####            data_dir: str,
#####            batch_size_per_gpu: int,
#####
#####            # specify the transform
#####            global_scale: Tuple[float, float],
#####            local_scale: Tuple[float, float],
#####            **kargs,
#####    ):
#####        super().__init__()
#####        print("-----> running datamodule init")
#####
#####        # params for overwriting the abstract property
#####        self._global_size = global_size
#####        self._local_size = local_size
#####        self._n_global_crops = n_global_crops
#####        self._n_local_crops = n_local_crops
#####
#####        # global params
#####        self.data_dir = data_dir
#####        self.batch_size_per_gpu = batch_size_per_gpu
#####        self.num_workers = cpu_count() if num_workers == -1 else num_workers
#####        self.gpus = torch.cuda.device_count()
#####
#####        # specify the transform
#####        self.global_scale = global_scale
#####        self.local_scale = local_scale
#####
#####    def get_metadata_to_regress(self, metadata: MetadataCropperDataset) -> Dict[str, float]:
#####        """ Extract one or more quantities to regress from the metadata """
#####        return dict()
#####
#####    def get_metadata_to_classify(self, metadata: MetadataCropperDataset) -> Dict[str, int]:
#####        """ Extract one or more quantities to classify from the metadata """
#####        return dict()
#####
#####    @classproperty
#####    def ch_in(self) -> int:
#####        return 3
#####
#####    @property
#####    def global_size(self) -> int:
#####        return self._global_size
#####
#####    @property
#####    def local_size(self) -> int:
#####        return self._local_size
#####
#####    @property
#####    def n_global_crops(self) -> int:
#####        return self._n_global_crops
#####
#####    @property
#####    def n_local_crops(self) -> int:
#####        return self._n_local_crops
#####
#####    @property
#####    def cropper_test(self):
#####        return CropperSparseTensor(
#####            strategy='identity',
#####            n_crops=1)
#####
#####    @property
#####    def trsfm_test(self) -> Callable:
#####        return TransformForList(
#####            transform_before_stack=torchvision.transforms.Compose([
#####                ToRgb(),
#####                LargestSquareCrop(self.global_size),
#####                RandomHFlip(p=0.5),
#####            ]),
#####            transform_after_stack=None,
#####        )
#####
#####    @property
#####    def cropper_train(self):
#####        return CropperSparseTensor(
#####            strategy='identity',
#####            n_crops=1)
#####
#####    @property
#####    def trsfm_train_global(self) -> Callable:
#####        return TransformForList(
#####            transform_before_stack=torchvision.transforms.Compose([
#####                ToRgb(),
#####                LargestSquareCrop(self.global_size),
#####            ]),
#####            transform_after_stack=torchvision.transforms.Compose([
#####                RandomHFlip(p=0.5),
#####                torchvision.transforms.RandomResizedCrop(
#####                    size=(self.global_size, self.global_size),
#####                    scale=self.global_scale,
#####                    ratio=(0.8, 1.2),
#####                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
#####            ])
#####        )
#####
#####    @property
#####    def trsfm_train_local(self) -> Callable:
#####        return TransformForList(
#####            transform_before_stack=torchvision.transforms.Compose([
#####                ToRgb(),
#####                LargestSquareCrop(self.global_size),
#####            ]),
#####            transform_after_stack=torchvision.transforms.Compose([
#####                RandomHFlip(p=0.5),
#####                torchvision.transforms.RandomResizedCrop(
#####                    size=(self.local_size, self.local_size),
#####                    scale=self.local_scale,
#####                    ratio=(0.8, 1.2),
#####                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
#####            ])
#####        )
#####
#####    @classmethod
#####    def add_specific_args(cls, parent_parser):
#####
#####        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')
#####
#####        # global parameters
#####        parser.add_argument("--data_dir", type=str, default="./caltech101_data",
#####                            help="directory where to download the data")
#####        parser.add_argument("--num_workers", default=cpu_count(), type=int,
#####                            help="number of worker to load data. Meaningful only if dataset is on disk. \
#####                            Set to zero if data in memory")
#####
#####        # specify the transform
#####        parser.add_argument("--global_size", type=int, default=128, help="size in pixel of the global crops")
#####        parser.add_argument("--local_size", type=int, default=64, help="size in pixel of the local crops")
#####        parser.add_argument("--global_scale", type=float, nargs=2, default=[0.7, 1.0],
#####                            help="scale used in RandomResizedCrop for the global crops")
#####        parser.add_argument("--local_scale", type=float, nargs=2, default=[0.3, 0.5],
#####                            help="scale used in RandomResizedCrop for the local crops")
#####        parser.add_argument("--n_local_crops", type=int, default=6, help="number of local crops")
#####        parser.add_argument("--n_global_crops", type=int, default=2, help="number of global crops")
#####
#####        # test dataset
#####        parser.add_argument("--batch_size_per_gpu", type=int, default=64,
#####                            help="Batch size FOR EACH GPUs. Set small for rapid prototyping. \
#####                            The total batch_size will increase linearly with the number of GPUs. \
#####                            If strategy is tiling then the actual batch_size might be smaller than this one")
#####
#####        return parser
#####
#####    def __tar_exists__(self):
#####        return exists(os.path.join(self.data_dir, "caltech101.tar.gz"))
#####
#####    def __download__tar__(self):
#####        from google.cloud import storage
#####
#####        bucket_name = "ld-data-bucket"
#####        source_blob_name = "tissue-purifier/caltech101.tar.gz"
#####        destination_file_name = os.path.join(self.data_dir, "caltech101.tar.gz")
#####
#####        # create the directory where the file will be written
#####        import os.path
#####        dirname = os.path.dirname(destination_file_name)
#####        os.makedirs(dirname, exist_ok=True)
#####
#####        # connect ot the google bucket
#####        storage_client = storage.Client()
#####        bucket = storage_client.bucket(bucket_name)
#####
#####        # Construct a client side representation of a blob.
#####        # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
#####        # any content from Google Cloud Storage. As we don't need additional data,
#####        # using `Bucket.blob` is preferred here.
#####        blob = bucket.blob(source_blob_name)
#####        blob.download_to_filename(destination_file_name)
#####
#####        print(
#####            "Downloaded storage object {} from bucket {} to local file {}.".format(
#####                source_blob_name, bucket_name, destination_file_name
#####            )
#####        )
#####
#####    def __untar__(self):
#####        if exists(os.path.join(self.data_dir, "caltech101")):
#####            print("untar already exists")
#####        else:
#####            tar_file_name = os.path.join(self.data_dir, "caltech101.tar.gz")
#####            # untar the tar.gz file
#####            with tarfile.open(tar_file_name, "r:gz") as fp:
#####                fp.extractall(path=self.data_dir)
#####
#####    def prepare_data(self):
#####        # this are thing to be done only once in distributed settings
#####        print("-----> running datamodule prepare_data")
#####        if self.__tar_exists__():
#####            print("untar data")
#####            self.__untar__()
#####        else:
#####            print("download and untar data")
#####            self.__download__tar__()
#####            self.__untar__()
#####
#####        # Image and category from folder dataset
#####        dataset_img_category = torchvision.datasets.ImageFolder(
#####            root=os.path.join(self.data_dir, 'caltech101/101_ObjectCategories'),
#####            transform=torchvision.transforms.ToTensor())
#####
#####        # prepare to split the data into train and test (test has 10 images for category)
#####        cat_id_for_each_img = torch.tensor([dataset_img_category.samples[n][1]
#####                                            for n in range(dataset_img_category.__len__())])
#####        counts_for_category = torch.bincount(cat_id_for_each_img)
#####        end_of_each_category = torch.cumsum(counts_for_category, dim=0)
#####        mask_train = torch.ones_like(cat_id_for_each_img).bool()
#####        for n in end_of_each_category:
#####            mask_train[n-10:n] = False
#####        indices_full = torch.arange(dataset_img_category.__len__())
#####        indices_train = indices_full[mask_train]
#####        indices_test = indices_full[~mask_train]
#####
#####        # Add the fake metadata and split into train and test dataset
#####        dataset_full = AddFakeMetadata(dataset_img_category)
#####        dataset_train = torch.utils.data.dataset.Subset(dataset=dataset_full, indices=indices_train)
#####        dataset_test = torch.utils.data.dataset.Subset(dataset=dataset_full, indices=indices_test)
#####
#####        print("train dataset length -->", dataset_train.__len__())
#####        print("test dataset length -->", dataset_test.__len__())
#####
#####        torch.save(dataset_train, os.path.join(self.data_dir, "dataset_train.pt"))
#####        torch.save(dataset_test, os.path.join(self.data_dir, "dataset_test.pt"))
#####
#####    def setup(self, stage: Optional[str] = None) -> None:
#####        """ stage: either 'fit', 'validate', 'test', or 'predict' """
#####        print("-----> running datamodule setup. stage ->{0}".format(stage))
#####        pass
#####
#####    def train_dataloader(self) -> DataLoaderWithLoad:
#####        dataset_train = torch.load(os.path.join(self.data_dir, "dataset_train.pt"))
#####
#####        dataloader_train = DataLoaderWithLoad(
#####            dataset_train,
#####            batch_size=min(dataset_train.__len__(), self.batch_size_per_gpu * max(1, self.gpus)),
#####            collate_fn=CollateFnListTuple(),
#####            num_workers=self.num_workers,
#####            shuffle=True,
#####            drop_last=True,
#####            pin_memory=False
#####        )
#####        return dataloader_train
#####
#####    def val_dataloader(self) -> List[DataLoaderWithLoad]:  # the same as test
#####        dataset_test = torch.load(os.path.join(self.data_dir, "dataset_test.pt"))
#####
#####        test_dataloader = DataLoaderWithLoad(
#####            dataset_test,
#####            batch_size=min(dataset_test.__len__(), self.batch_size_per_gpu * max(1, self.gpus)),
#####            collate_fn=CollateFnListTuple(),
#####            num_workers=self.num_workers,
#####            shuffle=False,
#####            drop_last=False,
#####            pin_memory=False,
#####        )
#####        return [test_dataloader]
#####
#####
#####class FashionDM(DinoDM):
#####    def __init__(
#####            self,
#####            # stuff to redefine abstract properties
#####            global_size: int,
#####            local_size: int,
#####            n_local_crops: int,
#####            n_global_crops: int,
#####
#####            # additional arguments
#####            num_workers: int,
#####            data_dir: str,
#####            batch_size_per_gpu: int,
#####
#####            # specify the transform
#####            global_scale: Tuple[float, float],
#####            local_scale: Tuple[float, float],
#####            **kargs,
#####    ):
#####        super().__init__()
#####        print("-----> running datamodule init")
#####
#####        # params for overwriting the abstract property
#####        self._global_size = global_size
#####        self._local_size = local_size
#####        self._n_global_crops = n_global_crops
#####        self._n_local_crops = n_local_crops
#####
#####        # global params
#####        self.data_dir = data_dir
#####        self.batch_size_per_gpu = batch_size_per_gpu
#####        self.num_workers = cpu_count() if num_workers == -1 else num_workers
#####        self.gpus = torch.cuda.device_count()
#####
#####        # specify the transform
#####        self.global_scale = global_scale
#####        self.local_scale = local_scale
#####
#####    def get_metadata_to_regress(self, metadata: MetadataCropperDataset) -> Dict[str, float]:
#####        """ Extract one or more quantities to regress from the metadata """
#####        return dict()
#####
#####    def get_metadata_to_classify(self, metadata: MetadataCropperDataset) -> Dict[str, int]:
#####        """ Extract one or more quantities to classify from the metadata """
#####        return dict()
#####
#####    @classproperty
#####    def ch_in(self) -> int:
#####        return 1
#####
#####    @property
#####    def global_size(self) -> int:
#####        return self._global_size
#####
#####    @property
#####    def local_size(self) -> int:
#####        return self._local_size
#####
#####    @property
#####    def n_global_crops(self) -> int:
#####        return self._n_global_crops
#####
#####    @property
#####    def n_local_crops(self) -> int:
#####        return self._n_local_crops
#####
#####    @property
#####    def cropper_test(self):
#####        return CropperSparseTensor(
#####            strategy='identity',
#####            n_crops=1)
#####
#####    @property
#####    def trsfm_test(self) -> Callable:
#####        return TransformForList(
#####            transform_before_stack=None,
#####            transform_after_stack=torchvision.transforms.Resize(size=self.global_size),
#####        )
#####
#####    @property
#####    def cropper_train(self):
#####        return CropperSparseTensor(
#####            strategy='identity',
#####            n_crops=1)
#####
#####    @property
#####    def trsfm_train_global(self) -> Callable:
#####        return TransformForList(
#####            transform_before_stack=None,
#####            transform_after_stack=torchvision.transforms.Resize(size=self.global_size),
#####        )
#####
#####    @property
#####    def trsfm_train_local(self) -> Callable:
#####        return TransformForList(
#####            transform_before_stack=None,
#####            transform_after_stack=torchvision.transforms.Resize(size=self.local_size),
#####        )
#####
#####    @classmethod
#####    def add_specific_args(cls, parent_parser):
#####
#####        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')
#####
#####        # global parameters
#####        parser.add_argument("--data_dir", type=str, default="./fashion_data",
#####                            help="directory where to download the data")
#####        parser.add_argument("--num_workers", default=cpu_count(), type=int,
#####                            help="number of worker to load data. Meaningful only if dataset is on disk. \
#####                            Set to zero if data in memory")
#####
#####        # specify the transform
#####        parser.add_argument("--global_size", type=int, default=32, help="size in pixel of the global crops")
#####        parser.add_argument("--local_size", type=int, default=32, help="size in pixel of the local crops")
#####        parser.add_argument("--global_scale", type=float, nargs=2, default=[0.7, 1.0],
#####                            help="scale used in RandomResizedCrop for the global crops")
#####        parser.add_argument("--local_scale", type=float, nargs=2, default=[0.3, 0.5],
#####                            help="scale used in RandomResizedCrop for the local crops")
#####        parser.add_argument("--n_local_crops", type=int, default=6, help="number of local crops")
#####        parser.add_argument("--n_global_crops", type=int, default=2, help="number of global crops")
#####
#####        # test dataset
#####        parser.add_argument("--batch_size_per_gpu", type=int, default=64,
#####                            help="Batch size FOR EACH GPUs. Set small for rapid prototyping. \
#####                            The total batch_size will increase linearly with the number of GPUs. \
#####                            If strategy is tiling then the actual batch_size might be smaller than this one")
#####
#####        return parser
#####
#####    def setup(self, stage: Optional[str] = None) -> None:
#####        """ stage: either 'fit', 'validate', 'test', or 'predict' """
#####        print("-----> running datamodule setup. stage ->{0}".format(stage))
#####        pass
#####
#####    def train_dataloader(self) -> DataLoaderWithLoad:
#####        fashion_train = torchvision.datasets.FashionMNIST(
#####            root="./fashion",
#####            train=True,
#####            transform=torchvision.transforms.ToTensor(),
#####            download=True)
#####
#####        dataset_train = AddFakeMetadata(fashion_train)
#####
#####        dataloader_train = DataLoaderWithLoad(
#####            dataset_train,
#####            batch_size=min(dataset_train.__len__(), self.batch_size_per_gpu * max(1, self.gpus)),
#####            collate_fn=CollateFnListTuple(),
#####            num_workers=self.num_workers,
#####            shuffle=True,
#####            drop_last=True,
#####            pin_memory=False
#####        )
#####        return dataloader_train
#####
#####    def val_dataloader(self) -> List[DataLoaderWithLoad]:  # the same as test
#####        fashion_test = torchvision.datasets.FashionMNIST(
#####            root="./fashion",
#####            train=False,
#####            transform=torchvision.transforms.ToTensor(),
#####            download=True)
#####
#####        dataset_test = AddFakeMetadata(fashion_test)
#####
#####        test_dataloader = DataLoaderWithLoad(
#####            dataset_test,
#####            batch_size=min(dataset_test.__len__(), self.batch_size_per_gpu * max(1, self.gpus)),
#####            collate_fn=CollateFnListTuple(),
#####            num_workers=self.num_workers,
#####            shuffle=False,
#####            drop_last=False,
#####            pin_memory=False,
#####        )
#####        return [test_dataloader]
#####
#####
#####