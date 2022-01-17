# integration tests
import pytest
from tissue_purifier.model_utils.dino import DinoModel
from tissue_purifier.model_utils.vae import VaeModel


# @pytest.mark.parametrize("hard_bootstrapping", [True, False])
def test_dino(dummy_dino_dm, trainer):
    datamodule_ch_in = dummy_dino_dm.ch_in

    config_dict = DinoModel.get_default_params()
    config_dict["image_in_ch"] = datamodule_ch_in

    dino_model = DinoModel(**config_dict)
    assert isinstance(dino_model, DinoModel)

    trainer.fit(model=dino_model, datamodule=dummy_dino_dm)


def test_vae(dummy_dino_dm, trainer):
    datamodule_ch_in = dummy_dino_dm.ch_in

    config_dict = VaeModel.get_default_params()
    config_dict["image_in_ch"] = datamodule_ch_in

    vae_model = VaeModel(**config_dict)
    assert isinstance(vae_model, VaeModel)

    trainer.fit(model=vae_model, datamodule=dummy_dino_dm)

