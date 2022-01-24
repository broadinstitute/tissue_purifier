import pytest
import torch
import torchvision
from torch.utils.data import DataLoader
from tissue_purifier.data_utils.sparse_image import SparseImage
from tissue_purifier.data_utils.dataset import CropperDataset
from tissue_purifier.data_utils.datamodule import (
    DummyDM,
    SlideSeqKidneyDM,
    SlideSeqTestisDM)


# @pytest.mark.parametrize("dm_type", ["dummy", "testis", "kidney"])
@pytest.mark.parametrize("dm_type", ["dummy"])
def test_dm(dm_type, capsys):
    if dm_type == 'dummy':
        dm = DummyDM()
    elif dm_type == 'testis':
        dm = SlideSeqTestisDM()
    elif dm_type == 'kidney':
        dm = SlideSeqKidneyDM(cohort='small')
    else:
        raise Exception("wrong DM type")

    dm.prepare_data()
    dm.setup(stage=None)

    # test the train_loader
    train_loader = dm.train_dataloader()
    assert isinstance(train_loader, DataLoader)

    batch = next(iter(train_loader))
    assert isinstance(batch, tuple)
    list_of_tensors = batch[0]
    assert isinstance(list_of_tensors, list)
    assert isinstance(list_of_tensors[0], torch.sparse.Tensor)

    train_dataset = train_loader.dataset
    assert isinstance(train_dataset, CropperDataset)

    sp_images = train_dataset.imgs
    assert isinstance(sp_images[0], SparseImage) or \
           isinstance(sp_images[0], torch.sparse.Tensor) or \
           isinstance(sp_images[0], torch.Tensor)
    assert sp_images[0].shape[-3] == dm.ch_in

    # test the val_loader
    test_loaders = dm.val_dataloader()
    assert isinstance(test_loaders, list)
    assert isinstance(test_loaders[0], DataLoader)

    batch = next(iter(test_loaders[0]))
    assert isinstance(batch, tuple)
    list_of_tensors = batch[0]
    assert isinstance(list_of_tensors, list)
    assert isinstance(list_of_tensors[0], torch.sparse.Tensor)

    test_dataset = test_loaders[0].dataset
    assert isinstance(test_dataset, CropperDataset)

    sp_images = test_dataset.imgs
    assert isinstance(sp_images[0], SparseImage) or \
           isinstance(sp_images[0], torch.sparse.Tensor) or \
           isinstance(sp_images[0], torch.Tensor)
    assert sp_images[0].shape[-3] == dm.ch_in

    # with capsys.disabled():
    #     print(dm.trsfm_train_global)
    #     print(dm.trsfm_train_local)
    #     print(dm.trsfm_test)
