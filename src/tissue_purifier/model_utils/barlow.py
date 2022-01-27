from typing import Sequence, List, Any, Dict

import torch
from argparse import ArgumentParser
import torchvision

from neptune.new.types import File
from tissue_purifier.model_utils.benckmark_model import BenchmarkModel
from tissue_purifier.data_utils.dataset import MetadataCropperDataset
from tissue_purifier.plot_utils.plot_images import show_raw_all_channels, show_corr_matrix
from tissue_purifier.misc_utils.misc import LARS
from tissue_purifier.misc_utils.dict_util import concatenate_list_of_dict
from tissue_purifier.misc_utils.misc import (
    smart_bool,
    linear_warmup_and_cosine_protocol)


def barlow_loss(
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        lambda_param: float = 5e-3,
        eps: float = 1e-5) -> (torch.Tensor, torch.Tensor):

    assert z_b.shape == z_a.shape
    batch_size, latent_dim = z_a.shape

    # normalize repr. along the batch dimension
    z_a_std, z_a_mean = torch.std_mean(z_a, unbiased=True, dim=-2)
    z_b_std, z_b_mean = torch.std_mean(z_b, unbiased=True, dim=-2)
    z_a_norm = (z_a - z_a_mean) / (z_a_std + eps)  # shape: BxD
    z_b_norm = (z_b - z_b_mean) / (z_b_std + eps)  # shape: BxD

    # cross-correlation matrix
    cross_corr = torch.mm(z_a_norm.T, z_b_norm) / batch_size  # shape: DxD

    # compute the loss
    mask_diag = torch.eye(latent_dim, dtype=torch.bool, device=cross_corr.device)
    on_diag = cross_corr[mask_diag].add_(-1).pow_(2).sum()
    off_diag = cross_corr[~mask_diag].pow_(2).sum()
    loss = on_diag + lambda_param * off_diag
    return loss, cross_corr.detach()


class BarlowModel(BenchmarkModel):
    """
    See
    official: https://github.com/facebookresearch/barlowtwins
    implementation for tiny datasets: https://github.com/IgorSusmelj/barlowtwins/blob/main/main.py
    """
    def __init__(
            self,
            # architecture
            backbone_type: str,
            backbone_pretrained: bool,
            image_in_ch: int,
            head_hidden_chs: List[int],
            head_out_ch: int,
            # loss
            lambda_off_diagonal: float,
            # optimizer
            optimizer_type: str,
            # scheduler
            warm_up_epochs: int,
            warm_down_epochs: int,
            max_epochs: int,
            min_learning_rate: float,
            max_learning_rate: float,
            min_weight_decay: float,
            max_weight_decay: float,
            # validation
            val_iomin_threshold: float = 0.0,
            **kwargs,
            ):
        super(BarlowModel, self).__init__(val_iomin_threshold=val_iomin_threshold)

        # Next two lines will make checkpointing much simpler. Always keep them as-is
        self.save_hyperparameters()  # all hyperparameters are saved to the checkpoint
        self.neptune_run_id = None  # if from scratch neptune_experiment_is is None

        # architecture
        self.backbone = self.init_backbone(
            backbone_pretrained=backbone_pretrained,
            backbone_in_ch=image_in_ch,
            backbone_type=backbone_type)

        tmp_in = torch.zeros((1, image_in_ch, 64, 64))
        tmp_out = self.backbone(tmp_in)
        backbone_ch_out = tmp_out.shape[-3]

        self.projection = self.init_projection(
            ch_in=backbone_ch_out,
            ch_hidden=head_hidden_chs,
            ch_out=head_out_ch)

        # loss
        self.lambda_off_diagonal = lambda_off_diagonal
        self.cross_corr = None

        # optimizer
        self.optimizer_type = optimizer_type

        # scheduler
        assert warm_up_epochs + warm_down_epochs <= max_epochs
        self.learning_rate_fn = linear_warmup_and_cosine_protocol(
            f_values=(min_learning_rate, max_learning_rate, min_learning_rate),
            x_milestones=(0, warm_up_epochs, max_epochs - warm_down_epochs, max_epochs))
        self.weight_decay_fn = linear_warmup_and_cosine_protocol(
            f_values=(min_weight_decay, min_weight_decay, max_weight_decay),
            x_milestones=(0, warm_up_epochs, max_epochs - warm_down_epochs, max_epochs))

        # default metadata to regress/classify is None
        self._get_metadata_to_classify = lambda x: dict()
        self._get_metadata_to_regress = lambda x: dict()

    @staticmethod
    def init_backbone(
            backbone_pretrained: bool,
            backbone_in_ch: int,
            backbone_type: str) -> torch.nn.Module:

        zero_init_residual = ~backbone_pretrained
        if backbone_type == 'resnet18':
            net = torchvision.models.resnet18(pretrained=backbone_pretrained, zero_init_residual=zero_init_residual)
        elif backbone_type == 'resnet34':
            net = torchvision.models.resnet34(pretrained=backbone_pretrained, zero_init_residual=zero_init_residual)
        elif backbone_type == 'resnet50':
            net = torchvision.models.resnet50(pretrained=backbone_pretrained, zero_init_residual=zero_init_residual)
        else:
            raise Exception("backbone_type not recognized. Received ->", backbone_type)

        first_conv_out_channels = list(net.children())[0].out_channels

        # Replace the first conv of Resnet, exclude the last fc layer. Keep the AdaptiveAvgPool2d
        backbone = torch.nn.Sequential(
            torch.nn.Conv2d(
                backbone_in_ch,
                first_conv_out_channels,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            ),
            *list(net.children())[1:-1],
        )
        return backbone

    @staticmethod
    def init_projection(
            ch_in: int,
            ch_out: int,
            ch_hidden: List[int]=None):

        sizes = [ch_in] + ch_hidden + [ch_out]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(torch.nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(torch.nn.BatchNorm1d(sizes[i + 1]))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(sizes[-2], sizes[-1], bias=False))
        return torch.nn.Sequential(*layers)

    def get_metadata_to_regress(self, metadata) -> dict:
        try:
            return self.trainer.datamodule.get_metadata_to_regress(metadata)
        except AttributeError:
            return self._get_metadata_to_regress(metadata)

    def get_metadata_to_classify(self, metadata) -> dict:
        try:
            return self.trainer.datamodule.get_metadata_to_classify(metadata)
        except AttributeError:
            return self._get_metadata_to_classify(metadata)

    @property
    def n_global_crops(self):
        return self.trainer.datamodule.n_global_crops

    @property
    def n_local_crops(self):
        return self.trainer.datamodule.n_local_crops

    @property
    def trsfm_train_local(self):
        return self.trainer.datamodule.trsfm_train_local

    @property
    def trsfm_train_global(self):
        return self.trainer.datamodule.trsfm_train_global

    @property
    def trsfm_test(self):
        return self.trainer.datamodule.trsfm_test

    @classmethod
    def add_specific_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')

        # validation
        parser.add_argument("--val_iomin_threshold", type=float, default=0.0,
                            help="during validation, only patches with IoMinArea < IoMin_threshold are used "
                                 "in the kn-classifier and kn-regressor.")

        # architecture
        parser.add_argument("--image_in_ch", type=int, default=3, help="number of channels in the input images")
        parser.add_argument("--backbone_type", type=str, default="resnet34", help="backbone type",
                            choices=['resnet18', 'resnet34', 'resnet50'])
        parser.add_argument("--backbone_pretrained", type=smart_bool, default=True, help="Use a pretrained backbone")
        parser.add_argument("--head_hidden_chs", type=int, nargs='+', default=[2048, 2048],
                            help="List of integers. Hidden channels in projection head.")
        parser.add_argument("--head_out_ch", type=int, default=2048, help="head output channels")

        # loss
        parser.add_argument("--lambda_off_diagonal", type=float, default=5e-3,
                            help="lambda multiplying off diagonal elements in the loss")

        # optimizer
        parser.add_argument("--optimizer_type", type=str, default='adam', help="optimizer type",
                            choices=['adamw', 'lars', 'sgd', 'adam', 'rmsprop'])

        # scheduler
        parser.add_argument("--max_epochs", default=1000, type=int,
                            help="Total number of epochs in training.")
        parser.add_argument("--warm_up_epochs", default=100, type=int,
                            help="Number of epochs for the linear learning-rate warm up.")
        parser.add_argument("--warm_down_epochs", default=500, type=int,
                            help="Number of epochs for the cosine decay.")
        parser.add_argument('--min_learning_rate', type=float, default=1e-5,
                            help="Target LR at the end of cosine protocol (smallest LR used during training).")
        parser.add_argument("--max_learning_rate", type=float, default=5e-4,
                            help="learning rate at the end of linear ramp (largest LR used during training).")
        parser.add_argument('--min_weight_decay', type=float, default=0.04,
                            help="Minimum value of the weight decay. It is used during the linear ramp.")
        parser.add_argument('--max_weight_decay', type=float, default=0.4,
                            help="Maximum Value of the weight decay. It is reached at the end of cosine protocol.")
        return parser

    @classmethod
    def get_default_params(cls) -> dict:
        parser = ArgumentParser()
        parser = BarlowModel.add_specific_args(parser)
        args = parser.parse_args(args=[])
        return args.__dict__

    def __log_example_images__(self, which_loaders: str, n_examples: int = 10, n_cols: int = 5):
        if which_loaders == "val":
            loaders = self.trainer.datamodule.val_dataloader()
            log_name = "val_imgs"
        elif which_loaders == "train":
            loaders = self.trainer.datamodule.train_dataloader()
            log_name = "train_imgs"
        elif which_loaders == "predict":
            loaders = self.trainer.datamodule.predict_dataloader()
            log_name = "predict_imgs"
        else:
            raise Exception("Invalid value for which_loaders. Expected 'val' or 'train' or 'predict'. \
            Received={0}".format(which_loaders))

        if not isinstance(loaders, Sequence):
            loaders = [loaders]

        for idx_dataloader, loader in enumerate(loaders):
            indeces = torch.randperm(n=loader.dataset.__len__())[:n_examples]
            list_imgs, _, _ = loader.load(index=indeces)
            list_imgs = list_imgs[:n_examples]

            tmp_ref = self.trsfm_test(list_imgs)
            tmp_ref_plot = show_raw_all_channels(tmp_ref, n_col=n_cols, show_colorbar=True)
            self.logger.run[log_name + "/ref_" + str(idx_dataloader)].log(File.as_image(tmp_ref_plot))

            if which_loaders == 'train':
                tmp_global = self.trsfm_train_global(list_imgs)
                tmp_global_plot = show_raw_all_channels(tmp_global, n_col=n_cols, show_colorbar=True)
                self.logger.run[log_name+"/global"].log(File.as_image(tmp_global_plot))

                tmp_local = self.trsfm_train_local(list_imgs)
                tmp_local_plot = show_raw_all_channels(tmp_local, n_col=n_cols, show_colorbar=True)
                self.logger.run[log_name + "/local"].log(File.as_image(tmp_local_plot))

    def on_predict_start(self) -> None:
        if self.global_rank == 0:
            self.__log_example_images__(n_examples=10, n_cols=5, which_loaders="predict")

    def on_validation_epoch_start(self) -> None:
        if self.global_rank == 0 and self.cross_corr is not None:
            fig_corr_matrix = show_corr_matrix(self.cross_corr, sup_title="epoch {0}".format(self.current_epoch))
            self.logger.run["corr_matrix"].log(File.as_image(fig_corr_matrix))

    def on_train_start(self) -> None:
        if self.global_rank == 0:
            self.__log_example_images__(n_examples=10, n_cols=5, which_loaders="train")

    def forward(self, x) -> (torch.Tensor, torch.Tensor):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection(y)
        return z, y

    def compute_features(self, x):
        with torch.no_grad():
            assert isinstance(x, torch.Tensor) and len(x.shape) == 4
            self.eval()
            z, y = self(x)
            self.train()
            return y

    def training_step(self, batch, batch_idx) -> dict:

        # this is data augmentation
        with torch.no_grad():
            list_imgs, list_labels, list_metadata = batch
            x_a = self.trsfm_train_global(list_imgs)
            x_b = self.trsfm_train_local(list_imgs)

        # forward is inside the no-grad context
        z_a, y_a = self(x_a)
        z_b, y_b = self(x_b)
        world_z_a, world_z_b = self.all_gather([z_a, z_b])
        world_z_a_flatten = world_z_a.flatten(end_dim=-2)
        world_z_b_flatten = world_z_b.flatten(end_dim=-2)
        batch_size_per_gpu = z_a.shape[0]
        batch_size_total = world_z_a_flatten.shape[0]

        loss, cross_corr = barlow_loss(
            z_a=world_z_a_flatten,
            z_b=world_z_b_flatten,
            lambda_param=self.lambda_off_diagonal,
            eps=1e-5)
        self.cross_corr = cross_corr

        # Update the optimizer parameters and log stuff
        with torch.no_grad():
            lr = self.learning_rate_fn(self.current_epoch)
            wd = self.weight_decay_fn(self.current_epoch)
            for i, pg in enumerate(self.optimizers().param_groups):
                pg["lr"] = lr
                if i == 0:  # only the first group is regularized
                    pg["weight_decay"] = wd
                else:
                    pg["weight_decay"] = 0.0

            # Finally I log interesting stuff
            self.log('train_loss', loss, on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
            self.log('train_loss', loss, on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
            self.log('weight_decay', wd, on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
            self.log('learning_rate', lr, on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
            self.log('batch_size_per_gpu_train', batch_size_per_gpu, on_step=False, on_epoch=True, rank_zero_only=True)
            self.log('batch_size_total_train', batch_size_total, on_step=False, on_epoch=True, rank_zero_only=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int = -1) -> Dict[str, torch.Tensor]:
        list_imgs: List[torch.sparse.Tensor]
        list_labels: List[int]
        list_metadata: List[MetadataCropperDataset]
        list_imgs, list_labels, list_metadata = batch

        # Compute the embeddings
        img = self.trsfm_test(list_imgs)
        z, y = self(img)

        # Collect the xywh for the patches in the validation
        w, h = img.shape[-2:]
        patch_x = torch.tensor([metadata.loc_x for metadata in list_metadata], dtype=z.dtype, device=z.device)
        patch_y = torch.tensor([metadata.loc_y for metadata in list_metadata], dtype=z.dtype, device=z.device)
        patch_w = w * torch.ones_like(patch_x)
        patch_h = h * torch.ones_like(patch_x)
        patches_xywh = torch.stack([patch_x, patch_y, patch_w, patch_h], dim=-1)

        # Create the validation dictionary. Note that all the entries are torch.Tensors
        val_dict = {
            "features_bbone": y,
            "features_head": z,
            "patches_xywh": patches_xywh
        }

        # Add to this dictionary the things I want to classify and regress
        dict_classify = concatenate_list_of_dict([self.get_metadata_to_classify(metadata)
                                                  for metadata in list_metadata])
        for k, v in dict_classify.items():
            val_dict["classify_"+k] = torch.tensor(v, device=self.device)

        dict_regress = concatenate_list_of_dict([self.get_metadata_to_regress(metadata)
                                                 for metadata in list_metadata])
        for k, v in dict_regress.items():
            val_dict["regress_" + k] = torch.tensor(v, device=self.device)

        return val_dict

    def configure_optimizers(self) -> torch.optim.Optimizer:
        regularized = []
        not_regularized = []
        for name, param in list(self.backbone.named_parameters()) + list(self.projection.named_parameters()):
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        arg_for_optimizer = [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.0}]

        # The real lr will be set in the training step
        # The weight_decay for the regularized group will be set in the training step
        if self.optimizer_type == 'adam':
            return torch.optim.Adam(arg_for_optimizer, betas=(0.9, 0.999), lr=0.0)
        elif self.optimizer_type == 'sgd':
            return torch.optim.SGD(arg_for_optimizer, momentum=0.9, lr=0.0)
        elif self.optimizer_type == 'rmsprop':
            return torch.optim.RMSprop(arg_for_optimizer, alpha=0.99, lr=0.0)
        elif self.optimizer_type == 'lars':
            # for convnet with large batch_size
            return LARS(arg_for_optimizer, momentum=0.9, lr=0.0)
        else:
            # do adamw
            raise Exception("optimizer is misspecified")

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """ Loading and resuming is handled automatically. Here I am dealing only with the special variables """
        self.neptune_run_id = checkpoint.get("neptune_run_id", None)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """ Loading and resuming is handled automatically. Here I am dealing only with the special variables """
        checkpoint["neptune_run_id"] = getattr(self.logger, "_run_short_id", None)
