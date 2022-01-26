from typing import Sequence, List, Any, Dict, Tuple, Callable

import torch
import numpy
from argparse import ArgumentParser
import torchvision

from torch.nn import functional as F
from neptune.new.types import File
from pytorch_lightning import LightningModule
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from tissue_purifier.plot_utils.plot_embeddings import plot_embeddings
from tissue_purifier.plot_utils.plot_images import show_batch
from tissue_purifier.model_utils.classify_regress import classify_and_regress
from tissue_purifier.misc_utils.nms import NonMaxSuppression

from tissue_purifier.misc_utils.dict_util import (
    concatenate_list_of_dict,
    subset_dict,
    subset_dict_non_overlapping_patches)

from tissue_purifier.misc_utils.misc import (
    smart_bool,
    SmartPca,
    SmartUmap,
    get_z_score,
    linear_warmup_and_cosine_protocol)


def make_encoder_backbone_from_resnet(in_channels: int, resnet_type: str):
    if resnet_type == 'resnet18':
        net = torchvision.models.resnet18(pretrained=True)
    elif resnet_type == 'resnet34':
        net = torchvision.models.resnet34(pretrained=True)
    elif resnet_type == 'resnet50':
        net = torchvision.models.resnet50(pretrained=True)
    else:
        raise Exception("Invalid enc_dec_type. Received {0}".format(resnet_type))

    first_conv_out_channels = list(net.children())[0].out_channels
    encoder_backbone = torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels,
            first_conv_out_channels,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        ),
        *list(net.children())[1:-2],  # note that I am excluding the last (fc) layer and the average_pool2D
    )
    return encoder_backbone


def make_decoder_backbone_from_resnet(resnet_type: str):
    from pl_bolts.models.autoencoders.components import DecoderBlock, ResNetDecoder, DecoderBottleneck

    if resnet_type == 'resnet18':
        net = ResNetDecoder(DecoderBlock, [2, 2, 2, 2], latent_dim=1,
                            input_height=1, first_conv=True, maxpool1=True)
        backbone_dec = torch.nn.Sequential(*list(net.children())[1:-3])  # remove the first and last layer
    elif resnet_type == 'resnet34':
        net = ResNetDecoder(DecoderBlock, [3, 4, 6, 3], latent_dim=1,
                            input_height=1, first_conv=True, maxpool1=True)
        backbone_dec = torch.nn.Sequential(*list(net.children())[1:-3])  # remove the first and last layer
    elif resnet_type == 'resnet50':
        net = ResNetDecoder(DecoderBottleneck, [3, 4, 6, 3], latent_dim=1,
                            input_height=1, first_conv=True, maxpool1=True)
        backbone_dec = torch.nn.Sequential(*list(net.children())[1:-3])  # remove the first and last layer
    else:
        raise NotImplementedError

    return backbone_dec


def make_encoder_backbone_from_scratch(in_channels: int, hidden_dims: Tuple[int]):
    modules = []
    ch_in = in_channels
    for h_dim in hidden_dims:
        modules.append(
            torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=ch_in,
                    out_channels=h_dim,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=1),
                torch.nn.BatchNorm2d(h_dim),
                torch.nn.LeakyReLU())
        )
        ch_in = h_dim
    encoder_backbone = torch.nn.Sequential(*modules)
    return encoder_backbone


def make_decoder_backbone_from_scratch(hidden_dims: Tuple[int]):
    modules = []

    for i in range(len(hidden_dims) - 1):
        modules.append(
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(
                    in_channels=hidden_dims[i],
                    out_channels=hidden_dims[i + 1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=1,
                    output_padding=1),
                torch.nn.BatchNorm2d(hidden_dims[i + 1]),
                torch.nn.LeakyReLU())
        )
    decoder_backbone = torch.nn.Sequential(*modules)
    return decoder_backbone


class ConvolutionalVae(torch.nn.Module):
    def __init__(self,
                 vae_type: str,
                 in_size: int,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: Tuple[int] = (32, 64, 128, 256, 512),
                 decoder_activation: torch.nn.Module = torch.nn.Identity(),
                 ) -> None:
        super(ConvolutionalVae, self).__init__()

        assert (in_size % 32) == 0, "The input size must be a multiple of 32. Received {0}".format(in_size)

        assert vae_type in ('vanilla', 'resnet18', 'resnet34', 'resnet50'), \
            "Invalid vae_type. Received {0}".format(vae_type)
        x_fake = torch.zeros((2, in_channels, in_size, in_size))

        # encoder
        self.latent_dim = latent_dim
        if vae_type == 'vanilla':
            self.encoder_backbone = make_encoder_backbone_from_scratch(in_channels=in_channels, hidden_dims=hidden_dims)
        elif vae_type.startswith("resnet"):
            self.encoder_backbone = make_encoder_backbone_from_resnet(in_channels=in_channels, resnet_type=vae_type)
        else:
            raise Exception("Invalid vae_type. Received {0}".format(vae_type))

        x_latent = self.encoder_backbone(x_fake)
        small_ch = x_latent.shape[-3]
        self.small_size = x_latent.shape[-1]
        self.fc_mu = torch.nn.Linear(small_ch * self.small_size * self.small_size, latent_dim)
        self.fc_var = torch.nn.Linear(small_ch * self.small_size * self.small_size, latent_dim)

        # Decoder
        self.decoder_input = torch.nn.Linear(latent_dim, small_ch * self.small_size * self.small_size)

        z_to_decode = torch.zeros((2, small_ch, self.small_size, self.small_size))
        if vae_type == 'vanilla':
            tmp_list = list(hidden_dims)
            tmp_list.reverse()
            reverse_hidden_dims = tuple(tmp_list)
            self.decoder_backbone = make_decoder_backbone_from_scratch(hidden_dims=reverse_hidden_dims)
        elif vae_type.startswith("resnet"):
            self.decoder_backbone = make_decoder_backbone_from_resnet(resnet_type=vae_type)
        else:
            raise Exception("Invalid vae_type. Received {0}".format(vae_type))

        x_tmp = self.decoder_backbone(z_to_decode)
        ch_tmp = x_tmp.shape[-3]
        last_hidden_ch = min(ch_tmp, 64)

        self.final_layer = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=ch_tmp,
                out_channels=last_hidden_ch,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=1,
                output_padding=1),
            torch.nn.BatchNorm2d(last_hidden_ch),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=last_hidden_ch,
                out_channels=in_channels,
                kernel_size=(3, 3),
                padding=1),
            decoder_activation,
        )

        # make sure the VAE reproduce the correct shape
        x_rec, x, mu, log_var = self.forward(x_fake)
        assert x_rec.shape == x_fake.shape

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.

        Args:
            x: (Tensor) [B x C x H x W]
            verbose: bool

        Returns:
            mu, log_var (Tensors) [B x latent_dim]
        """
        result = self.encoder_backbone(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        Args:
            z: (Tensor) [B x D]
            verbose: bool

        Returns:
            (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(z.shape[0], -1, self.small_size, self.small_size)
        result = self.decoder_backbone(result)
        result = self.final_layer(result)
        return result

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_rec = self.decode(z)
        return [x_rec, x, mu, log_var]


class VaeModel(LightningModule):
    def __init__(
            self,

            # architecture
            vae_type: str,
            image_size: int,
            image_in_ch: int,
            latent_dim: int,
            encoder_hidden_dims: Tuple[int],
            decoder_output_activation: str,

            # optimizer
            optimizer_type: str,
            beta_vae_init: float,
            momentum_beta_vae: float,

            # scheduler
            warm_up_epochs: int,
            warm_down_epochs: int,
            min_learning_rate: float,
            max_learning_rate: float,
            min_weight_decay: float,
            max_weight_decay: float,

            # gradient clipping (these parameters are defined)
            gradient_clip_val: float = 0.0,
            gradient_clip_algorithm: str = 'value',

            # validation
            val_iomin_threshold: float = 0.0,
            **kwargs,
            ):
        super().__init__()

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        # Next two lines will make checkpointing much simpler. Always keep them as-is
        self.save_hyperparameters()  # all hyperparameters are saved to the checkpoint
        self.neptune_run_id = None  # if from scratch neptune_run_id is None

        # to make sure that you load the inoput images only once
        self.already_loaded_input_val_images = False

        # validation
        self.val_iomin_threshold = val_iomin_threshold

        # architecture
        if decoder_output_activation == 'identity':
            output_activation = torch.nn.Identity()
        elif decoder_output_activation == 'relu':
            output_activation = torch.nn.ReLU()
        elif decoder_output_activation == 'tanh':
            output_activation = torch.nn.Tanh()
        elif decoder_output_activation == 'softplus':
            output_activation = torch.nn.Softplus()
        elif decoder_output_activation == "sigmoid":
            output_activation = torch.nn.Sigmoid()
        else:
            raise Exception("invalid decoder_output_activation. Received {0}".format(decoder_output_activation))

        self.image_size = image_size
        self.vae = ConvolutionalVae(
            vae_type=vae_type,
            in_size=image_size,
            in_channels=image_in_ch,
            latent_dim=latent_dim,
            hidden_dims=tuple(encoder_hidden_dims),
            decoder_activation=output_activation
        )

        # stuff to do gradient clipping internally
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm

        # stuff to keep the gradients and adjust beta_vae
        self.loss_type = None
        self.grad_due_to_kl = None
        self.grad_due_to_mse = None
        self.grad_old = None
        assert 0.0 < beta_vae_init < 1.0, \
            "Error. beta_vae_init should be in (0,1). Received {0}".format(beta_vae_init)
        self.register_buffer("beta_vae", float(beta_vae_init) * torch.ones(1, requires_grad=False).float())
        self.momentum_beta_vae = momentum_beta_vae

        # optimizer
        self.optimizer_type = optimizer_type
        self.learning_rate_fn = linear_warmup_and_cosine_protocol(
            f_values=(min_learning_rate, max_learning_rate, min_learning_rate),
            x_milestones=(0, warm_up_epochs, warm_up_epochs, warm_up_epochs + warm_down_epochs))
        self.weight_decay_fn = linear_warmup_and_cosine_protocol(
            f_values=(min_weight_decay, min_weight_decay, max_weight_decay),
            x_milestones=(0, warm_up_epochs, warm_up_epochs, warm_up_epochs + warm_down_epochs))

        # default metadata to regress/classify is None
        self._get_metadata_to_classify = lambda x: dict()
        self._get_metadata_to_regress = lambda x: dict()

    def get_metadata_to_regress(self, metadata) -> dict:
        try:
            return self.trainer.datamodule.get_metadata_to_regress(metadata)
        except AttributeError:
            return self._get_metadata_to_regress(metadata)

    def get_metadata_to_classify(self, metadata):
        try:
            return self.trainer.datamodule.get_metadata_to_classify(metadata)
        except AttributeError:
            return self._get_metadata_to_classify(metadata)

    @property
    def n_global_crops(self):
        return self.trainer.datamodule.n_global_crops

    @property
    def trsfm_train_global(self):
        return self.trainer.datamodule.trsfm_train_global

    @property
    def trsfm_train_local(self):
        return self.trainer.datamodule.trsfm_train_local

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

        # this model has manual optimization therefore it has to handle clipping internally.
        parser.add_argument("--gradient_clip_val", type=float, default=0.5,
                            help="Clip the gradients to this value. If 0 no clipping")
        parser.add_argument("--gradient_clip_algorithm", type=str, default="value", choices=["norm", "value"],
                            help="Algorithm to use for gradient clipping.")

        # architecture
        parser.add_argument("--vae_type", type=str, default="resnet18",
                            choices=["vanilla", "resnet18", "resnet34", "resnet50"],
                            help="The backbone architecture of the VAE")
        parser.add_argument("--image_size", type=int, default=64,
                            help="size in pixel of the input image. Must be a multiple of 32")
        parser.add_argument("--image_in_ch", type=int, default=3, help="number of channels of the input image")
        parser.add_argument("--latent_dim", type=int, default=128, help="number of latent dimensions")
        parser.add_argument("--encoder_hidden_dims", type=int, nargs='*', default=[32, 64, 128, 256, 512],
                            help="dimension of the hidden layers. Used only in vae_type='vanilla'.")
        parser.add_argument("--decoder_output_activation", type=str, default="identity",
                            choices=["sigmoid", "identity", "tanh", "softplus", "relu"],
                            help="The non-linearity used to produce the reconstructed image.")

        # optimizer
        parser.add_argument("--optimizer_type", type=str, default='adam', help="optimizer type",
                            choices=['adamw', 'sgd', 'adam', 'rmsprop'])

        # Parameters to update the beta (i.e. the balancing between MSE and KL)
        parser.add_argument('--beta_vae_init', type=float, default=0.1,
                            help="Initial value for beta (coefficient in front of KL). Should be in (0.0, 1.0)")
        parser.add_argument('--momentum_beta_vae', type=float, default=0.999,
                            help="momentum for the EMA which updates the value of beta")

        # scheduler
        parser.add_argument("--warm_up_epochs", default=100, type=int,
                            help="Number of epochs for the linear learning-rate warm up.")
        parser.add_argument("--warm_down_epochs", default=1000, type=int,
                            help="Number of epochs for the cosine decay.")
        parser.add_argument('--min_learning_rate', type=float, default=1e-5,
                            help="Target LR at the end of cosine protocol (smallest LR used during training).")
        parser.add_argument("--max_learning_rate", type=float, default=5e-4,
                            help="learning rate at the end of linear ramp (largest LR used during training).")
        parser.add_argument('--min_weight_decay', type=float, default=0.0,
                            help="Minimum value of the weight decay. It is used during the linear ramp.")
        parser.add_argument('--max_weight_decay', type=float, default=0.0,
                            help="Maximum Value of the weight decay. It is reached at the end of cosine protocol.")

        return parser

    @classmethod
    def get_default_params(cls) -> dict:
        parser = ArgumentParser()
        parser = cls.add_specific_args(parser)
        args = parser.parse_args(args=[])
        return args.__dict__

    def on_predict_start(self) -> None:
        if self.global_rank == 0:
            self.__log_example_images__(n_examples=10, n_cols=5, which_loaders="predict")

    def on_train_start(self) -> None:
        if self.global_rank == 0:
            self.__log_example_images__(n_examples=10, n_cols=5, which_loaders="train")
            # path_test = os_join(self.trainer.datamodule.data_dir, "train_dataset.pt")
            # path_train = os_join(self.trainer.datamodule.data_dir, "train_dataset.pt")
            # self.logger.run["dataset/test"].upload(path_test)
            # self.logger.run["dataset/train"].upload(path_train)

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
            tmp_ref_plot = show_batch(tmp_ref.float(), n_col=n_cols)
            # self.logger.log_image(log_name=log_name+"/ref_" + str(idx_dataloader), image=tmp_ref_plot)
            self.logger.run[log_name+"/ref_" + str(idx_dataloader)].log(File.as_image(tmp_ref_plot))

            if which_loaders == 'train':
                tmp_global = self.trsfm_train_global(list_imgs)
                tmp_global_plot = show_batch(tmp_global.float(), n_col=n_cols)
                # self.logger.log_image(log_name=log_name+"/global", image=tmp_global_plot)
                self.logger.run[log_name + "/global"].log(File.as_image(tmp_global_plot))

                tmp_local = self.trsfm_train_local(list_imgs)
                tmp_local_plot = show_batch(tmp_local.float(), n_col=n_cols)
                # self.logger.log_image(log_name=log_name + "/local", image=tmp_local_plot)
                self.logger.run[log_name + "/local"].log(File.as_image(tmp_local_plot))

    def compute_losses(self, x, x_rec, mu, log_var):
        # compute both kl and derivative of kl w.r.t. mu and log_var
        assert len(mu.shape) == 2
        batch_size = mu.shape[0]
        kl_loss = 0.5 * (mu ** 2 + log_var.exp() - log_var - 1.0).sum() / batch_size
        mse_loss = F.mse_loss(x, x_rec, reduction='mean')

        return {
            'mse_loss': mse_loss,
            'kl_loss': kl_loss,
        }

    def forward(self, x) -> (torch.Tensor, torch.Tensor):
        return self.vae(x)

    def compute_features(self, x):
        with torch.no_grad():
            assert isinstance(x, torch.Tensor) and len(x.shape) == 4
            self.eval()
            x_rec, x, mu, log_var = self.vae(x)
            self.train()
            return mu

    def training_step(self, batch, batch_idx):

        with torch.no_grad():
            # Update the optimizer parameters
            opt = self.optimizers()
            assert isinstance(opt, torch.optim.Optimizer)
            lr = self.learning_rate_fn(self.current_epoch)
            wd = self.weight_decay_fn(self.current_epoch)
            for i, param_group in enumerate(opt.param_groups):
                param_group["lr"] = lr
                param_group["weight_decay"] = wd

            # this is data augmentation
            list_imgs = batch[0]
            batch_size = len(list_imgs)
            img_in = self.trsfm_train_global(list_imgs)
            assert img_in.shape[-1] == self.image_size, \
                "img.shape {0} vs image_size {1}".format(img_in.shape[-1], self.image_size)

        # does the encoding-decoding
        x_rec, x, mu, log_var = self(img_in)
        loss_dict = self.compute_losses(x=x, x_rec=x_rec, mu=mu, log_var=log_var)

        # Manual optimization
        opt.zero_grad()
        loss_kl = self.beta_vae * loss_dict["kl_loss"]
        loss_mse = (1.0 - self.beta_vae) * loss_dict["mse_loss"]

        if batch_idx == 0:
            # two backward passes to collect the two gradients separately
            self.manual_backward(loss_kl, retain_graph=True)
            grad_due_to_kl_tmp = self.__get_grad_from_last_layer_of_encoder__()
            self.manual_backward(loss_mse, retain_graph=False)
            grad_tot_tmp = self.__get_grad_from_last_layer_of_encoder__()
            grad_due_to_mse_tmp = grad_tot_tmp - grad_due_to_kl_tmp
        else:
            # a single backward pass
            self.manual_backward(loss_kl + loss_mse)
            grad_due_to_kl_tmp, grad_due_to_mse_tmp = None, None

        self.clip_gradients(
            opt,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm=self.gradient_clip_algorithm
        )
        opt.step()
        # end manual optimization

        with torch.no_grad():
            self.log('weight_decay', wd,
                     on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
            self.log('learning_rate', lr,
                     on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)

            # Use the 75% quantile, i.e. we are requiring that 75% of the pixel are reconstructed better than rec_target
            mse_for_constraint = torch.quantile((x - x_rec).pow(2).sum(dim=-3), q=0.75)

            self.log('train_loss', loss_kl + loss_mse,
                     on_step=False, on_epoch=True, rank_zero_only=True, batch_size=batch_size)
            self.log('train_mse_loss', loss_dict['mse_loss'],
                     on_step=False, on_epoch=True, rank_zero_only=True, batch_size=batch_size)
            self.log('train_kl_loss', loss_dict['kl_loss'],
                     on_step=False, on_epoch=True, rank_zero_only=True, batch_size=batch_size)
            self.log('mse_for_constraint', mse_for_constraint,
                     on_step=False, on_epoch=True, rank_zero_only=True, batch_size=batch_size)

            # batch_size
            tmp_local_batch_size = torch.tensor(len(list_imgs), device=self.device, dtype=torch.float)
            world_batch_size = self.all_gather(tmp_local_batch_size)
            self.log('batch_size_per_gpu_train', world_batch_size.float().mean(),
                     on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
            self.log('batch_size_total_train', world_batch_size.float().sum(),
                     on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)

            # update the beta_vae if necessary
            if grad_due_to_mse_tmp is not None and grad_due_to_kl_tmp is not None:
                world_grad_due_to_mse, world_grad_due_to_kl = self.all_gather([grad_due_to_mse_tmp, grad_due_to_kl_tmp])
                if len(world_grad_due_to_mse.shape) == 1+len(grad_due_to_mse_tmp.shape):
                    grad_due_to_mse = world_grad_due_to_mse.mean(dim=0)
                    grad_due_to_kl = world_grad_due_to_kl.mean(dim=0)
                else:
                    grad_due_to_mse = world_grad_due_to_mse
                    grad_due_to_kl = world_grad_due_to_kl

                # print("grad_due_to_mse_tmp.shape", grad_due_to_mse_tmp.shape)
                # print("world_grad_due_to_mse.shape", world_grad_due_to_mse.shape)
                # print("grad_due_to_mse.shape", grad_due_to_mse.shape)

                c11 = torch.dot(grad_due_to_kl, grad_due_to_kl) / self.beta_vae**2
                c22 = torch.dot(grad_due_to_mse, grad_due_to_mse) / (1.0 - self.beta_vae)**2
                c12 = torch.dot(grad_due_to_kl, grad_due_to_mse) / (self.beta_vae * (1.0 - self.beta_vae))

                method = 0
                if method == 0:
                    # find beta in (0,1) which minimizes: || beta * grad_kl + (1-beta) * grad_mse ||^2
                    # see paper: "Multi-Task Learning as Multi-Objective Optimization"
                    # This is the close form solution
                    ideal_beta_vae = ((c22 - c12) / (c11 + c22 - 2 * c12)).clamp(min=0.0, max=1.0)
                elif method == 1:
                    # find beta in (0,1) which makes the two gradient equal size, i.e.:
                    # set: beta * sqrt(c11) = (1 - beta) * sqrt(c22)
                    # leads to: beta = sqrt(c22) / (sqrt(c11) + sqrt(c22))
                    ideal_beta_vae = (c22.sqrt() / (c11.sqrt() + c22.sqrt())).clamp(min=0.0, max=1.0)
                else:
                    raise Exception("Method can only be 0 or 1. Received {0}".format(method))

                # update beta using a slow Exponential Moving Average (EMA)
                self.__update_beta_vae__(ideal_beta=ideal_beta_vae, beta_momentum=self.momentum_beta_vae)

                self.log('beta/c11', c11, on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
                self.log('beta/c12', c12, on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
                self.log('beta/c22', c22, on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
                self.log('beta/beta_vae', self.beta_vae,
                         on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
                self.log('beta/ideal_beta_vae', ideal_beta_vae,
                         on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)

    def __get_grad_from_last_layer_of_encoder__(self) -> torch.Tensor:
        grad = torch.cat((
            self.vae.fc_mu.bias.grad.detach().clone().flatten(),
            self.vae.fc_mu.weight.grad.detach().clone().flatten(),
            self.vae.fc_var.bias.grad.detach().clone().flatten(),
            self.vae.fc_var.weight.grad.detach().clone().flatten()
        ), dim=0)
        return grad

    def __update_beta_vae__(self, ideal_beta, beta_momentum):
        # update only if the suggested beta is finite
        if ideal_beta.isfinite():
            tmp = beta_momentum * self.beta_vae + (1.0 - beta_momentum) * ideal_beta
            self.beta_vae = tmp.clamp(min=1.0E-5, max=1.0 - 1.0E-5)

    def validation_step(self, batch, batch_idx, dataloader_idx: int = -1) -> Dict[str, torch.Tensor]:
        list_imgs, list_labels, list_metadata = batch

        # Compute the embeddings
        img = self.trsfm_test(list_imgs)
        x_rec, x, mu, log_var = self(img)
        loss_dict = self.compute_losses(x=x, x_rec=x_rec, mu=mu, log_var=log_var)
        loss = self.beta_vae * loss_dict["kl_loss"] + (1.0 - self.beta_vae) * loss_dict["mse_loss"]

        batch_size = len(list_imgs)
        self.log('val_loss', loss, on_step=False, on_epoch=True, rank_zero_only=True,
                 batch_size=batch_size)
        self.log('val_mse_loss', loss_dict["mse_loss"], on_step=False, on_epoch=True, rank_zero_only=True,
                 batch_size=batch_size)
        self.log('val_kl_loss', loss_dict["kl_loss"], on_step=False, on_epoch=True, rank_zero_only=True,
                 batch_size=batch_size)

        if self.global_rank == 0 and batch_idx == 0:

            img_out = x_rec.clone().detach().float()  # make sure this is in full precision for plotting
            one_ch_tmp_out_plot = show_batch(img_out[0].unsqueeze(dim=-3), n_col=5,
                                             title="output, epoch={0}".format(self.current_epoch))
            self.logger.run["rec/output_imgs/one_ch"].log(File.as_image(one_ch_tmp_out_plot))
            all_ch_tmp_out_plot = show_batch(img_out[:10], n_col=5,
                                             title="output, epoch={0}".format(self.current_epoch))
            self.logger.run["rec/output_imgs/all_ch"].log(File.as_image(all_ch_tmp_out_plot))

            if not self.already_loaded_input_val_images:
                img_in = img.clone().detach().float()  # make sure this is in full precision for plotting
                one_ch_tmp_in_plot = show_batch(img_in[0].unsqueeze(dim=-3), n_col=5,
                                                title="input, epoch={0}".format(self.current_epoch))
                self.logger.run["rec/input_imgs/one_ch"].log(File.as_image(one_ch_tmp_in_plot))
                all_ch_tmp_in_plot = show_batch(img_in[:10], n_col=5,
                                                title="input, epoch={0}".format(self.current_epoch))
                self.logger.run["rec/input_imgs/all_ch"].log(File.as_image(all_ch_tmp_in_plot))
                self.already_loaded_input_val_images = True

        # Collect the xywh for the patches in the validation
        w, h = img.shape[-2:]
        patch_x = torch.tensor([metadata.loc_x for metadata in list_metadata], dtype=mu.dtype, device=mu.device)
        patch_y = torch.tensor([metadata.loc_y for metadata in list_metadata], dtype=mu.dtype, device=mu.device)
        patch_w = w * torch.ones_like(patch_x)
        patch_h = h * torch.ones_like(patch_x)
        patches_xywh = torch.stack([patch_x, patch_y, patch_w, patch_h], dim=-1)

        # Create the validation dictionary. Note that all the entries are torch.Tensors
        val_dict = {
            "features_mu": mu,
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

    def validation_epoch_end(self, list_of_val_dict) -> None:
        """ You can receive a list_of_valdict or, if you have multiple val_datasets, a list_of_list_of_valdict """
        print("--- inside validation epoch end")

        """ You can receive a list_of_valdict or, if you have multiple val_datasets, a list_of_list_of_valdict """
        print("inside validation epoch end")

        if isinstance(list_of_val_dict[0], dict):
            list_dict = [concatenate_list_of_dict(list_of_val_dict)]
        elif isinstance(list_of_val_dict[0], list):
            list_dict = [concatenate_list_of_dict(tmp_list) for tmp_list in list_of_val_dict]
        else:
            raise Exception("In validation epoch end. I received an unexpected input")

        for loader_idx, total_dict in enumerate(list_dict):
            print("rank {0} dataloader_idx {1}".format(self.global_rank, loader_idx))

            # gather dictionaries from the other processes and flatten the extra dimension.
            world_dict = self.all_gather(total_dict)
            all_keys = list(world_dict.keys())
            for key in all_keys:
                if len(world_dict[key].shape) == 1 + len(total_dict[key].shape):
                    # In interactive mode the all_gather does not add an extra leading dimension
                    # This check makes sure that I am not flattening when leading dim has not been added
                    world_dict[key] = world_dict[key].flatten(end_dim=1)
                # Add the z_score
                if key.startswith("feature") and key.endswith("bbone"):
                    tmp_key = key + '_zscore'
                    world_dict[tmp_key] = get_z_score(world_dict[key], dim=-2)
            print("done dictionary. rank {0}".format(self.global_rank))

            # DO operations ONLY on rank 0.
            # ADD "rank_zero_only=True" to avoid deadlocks on synchronization.
            if self.global_rank == 0:

                # plot the UMAP colored by all available annotations
                smart_pca = SmartPca(preprocess_strategy='z_score')
                smart_umap = SmartUmap(n_neighbors=25, preprocess_strategy='raw',
                                       n_components=2, min_dist=0.5, metric='euclidean')

                # plot the UMAP colored by all available annotations
                smart_pca = SmartPca(preprocess_strategy='z_score')
                smart_umap = SmartUmap(n_neighbors=25, preprocess_strategy='raw',
                                       n_components=2, min_dist=0.5, metric='euclidean')

                embedding_keys = ["features_mu"]
                for k in embedding_keys:
                    input_features = world_dict[k]
                    embeddings_pca = smart_pca.fit_transform(input_features, n_components=0.95)
                    embeddings_umap = smart_umap.fit_transform(embeddings_pca)
                    world_dict['pca_' + k] = embeddings_pca
                    world_dict['umap_' + k] = embeddings_umap

                annotation_keys, titles = [], []
                for k in world_dict.keys():
                    if k.startswith("regress") or k.startswith("classify"):
                        annotation_keys.append(k)
                        titles.append("{0} -> Epoch={1}".format(k, self.current_epoch))

                # Now I have both annotation_keys and feature_keys
                all_figs = []
                for embedding_key in embedding_keys:
                    fig_tmp = plot_embeddings(
                        world_dict,
                        embedding_key=embedding_key,
                        annotation_keys=annotation_keys,
                        x_label="UMAP1",
                        y_label="UMAP2",
                        titles=titles,
                        n_col=2,
                    )
                    all_figs.append(fig_tmp)

                for fig_tmp, key_tmp in zip(all_figs, embedding_keys):
                    self.logger.run["maps/" + key_tmp].log(File.as_image(fig_tmp))

                # Do KNN classification
                def exclude_self(d):
                    w = numpy.ones_like(d)
                    w[d == 0.0] = 0.0
                    return w

                kn_kargs = {
                    "n_neighbors": 5,
                    "weights": exclude_self,
                }

                feature_keys = ["features_mu"]
                regress_keys, classify_keys = [], []
                for key in world_dict.keys():
                    if key.startswith("regress"):
                        regress_keys.append(key)
                    elif key.startswith("classify"):
                        classify_keys.append(key)
                    elif key.startswith("pca_") or key.startswith("umap_"):
                        feature_keys.append(key)

                regressor = KNeighborsRegressor(**kn_kargs)
                classifier = KNeighborsClassifier(**kn_kargs)

                # loop over subset made of not-overlapping patches
                df_tot = None

                # compute the patch_to_patch overlap just one at the beginning
                patches = world_dict["patches_xywh"]
                initial_score = torch.rand_like(patches[:, 0].float())
                tissue_ids = world_dict["classify_tissue_label"]
                nms_mask_n, overlap_nn = NonMaxSuppression.compute_nm_mask(
                    score=initial_score,
                    ids=tissue_ids,
                    patches_xywh=patches,
                    iom_threshold=self.val_iomin_threshold)
                binarized_overlap_nn = (overlap_nn > self.val_iomin_threshold).float()

                for n in range(20):
                    print("loop over non-overlapping", n)
                    # create a dictionary with only non-overlapping patches to test kn-regressor/classifier
                    nms_mask_n = NonMaxSuppression._perform_nms_selection(mask_overlap_nn=binarized_overlap_nn,
                                                                          score_n=torch.rand_like(initial_score),
                                                                          possible_n=torch.ones_like(initial_score).bool())
                    world_dict_subset = subset_dict(input_dict=world_dict, mask=nms_mask_n)

                    df_tmp = classify_and_regress(
                        input_dict=world_dict_subset,
                        feature_keys=feature_keys,
                        regress_keys=regress_keys,
                        classify_keys=classify_keys,
                        regressor=regressor,
                        classifier=classifier,
                        n_repeats=1,
                        n_splits=1,
                        verbose=False,
                    )
                    df_tot = df_tmp if df_tot is None else df_tot.merge(df_tmp, how='outer')

                df_tot["combined_key"] = df_tot["x_key"] + "_" + df_tot["y_key"]
                df_mean = df_tot.groupby("combined_key").mean()
                df_std = df_tot.groupby("combined_key").std()

                for row in df_mean.itertuples():
                    for k, v in row._asdict().items():
                        if isinstance(v, float) and numpy.isfinite(v):
                            name = "kn/" + row.Index + "/" + k + "/mean"
                            self.log(name=name, value=v, batch_size=1, rank_zero_only=True)

                for row in df_std.itertuples():
                    for k, v in row._asdict().items():
                        if isinstance(v, float) and numpy.isfinite(v):
                            name = "kn/" + row.Index + "/" + k + "/std"
                            self.log(name=name, value=v, batch_size=1, rank_zero_only=True)

    def configure_optimizers(self):
        # the learning_rate and weight_decay are very large. They are just placeholder.
        # The real value will be set by the scheduler
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                self.vae.parameters(),
                lr=1000.0,
                weight_decay=1000.0)
        else:
            raise NotImplementedError

        return optimizer

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """ Loading and resuming is handled automatically. Here I am dealing only with the special variables """
        self.neptune_run_id = checkpoint.get("neptune_run_id", None)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """ Loading and resuming is handled automatically. Here I am dealing only with the special variables """
        checkpoint["neptune_run_id"] = getattr(self.logger, "_run_short_id", None)
