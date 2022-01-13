import pytest
import torch
import torchvision
from tissue_purifier.data_utils.sparse_image import SparseImage
from tissue_purifier.data_utils.dataset import CropperDataset
from tissue_purifier.data_utils.datamodule import DummyDM
from torch.utils.data import DataLoader


# TODO: make it run in a different folder 
def test_dummy_dm():
    dm = DummyDM()
    dm.prepare_data()
    dm.setup(stage=None)

    train_loader = dm.train_dataloader()
    assert isinstance(train_loader, DataLoader)

    train_dataset = train_loader.dataset
    assert isinstance(train_dataset, CropperDataset)

    sp_images = train_dataset.imgs
    assert isinstance(sp_images[0], SparseImage)

    # metadatas = train_dataset.metadatas
    # f_names = [meta.f_name for meta in metadatas]
    # cell_to_code_dict = sp_images[0]._categories_to_codes
    # cell_types = list(cell_to_code_dict.keys())
