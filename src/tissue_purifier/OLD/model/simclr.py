from typing import List, Any, Dict, Optional, Callable, Tuple

import torch
import numpy
from torch.nn import functional as F
from datetime import timedelta
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from tissue_purifier.data_utils.dataset import MetadataCropperDataset
from tissue_purifier.model_utils.misc import linear_warmup_and_decay, concatenate_list_of_dict


class NTXentLoss(torch.nn.Module):
    """ Very smart implementation of contrastive loss """
    def __init__(self,
                 temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        self.eps = 1e-8

    def forward(self,
                out0: torch.Tensor,
                out1: torch.Tensor,
                verbose: bool = False,
                ):
        """Forward pass through Contrastive Cross-Entropy Loss.

            Args:
                out0:
                    Output projections of the first set of transformed images.
                    Shape: (batch_size, embedding_size)
                out1:
                    Output projections of the second set of transformed images.
                    Shape: (batch_size, embedding_size)
                verbose: Print some information while running

            Returns:
                Contrastive Cross Entropy Loss value.

            Example:
            >>> batch, latent_dim = 3, 1028
            >>> out_1 = torch.randn((batch, latent_dim))
            >>> out_2 = out1 + 0.1  # this mimic a very good encoding where pair images have close embeddings
            >>> ntx_loss = NTXentLoss()
            >>> _ = ntx_loss(out_1, out_2, verbose=True)
        """

        device = out0.device
        batch_size, _ = out0.shape

        # normalize the output to length 1
        out0 = F.normalize(out0, dim=1)  # shape: batch_size, latent_dim
        out1 = F.normalize(out1, dim=1)  # shape: batch_size, latent_dim

        # use other samples from batch as negatives
        output = torch.cat((out0, out1), dim=0)  # shape: 2*batch_size, latent_dim

        # the logits are the similarity matrix divided by the temperature
        logits = torch.einsum('nc,mc->nm', output, output) / self.temperature  # shape: 2*batch_size, 2*batch_size

        if verbose:
            print("logits.shape", logits.shape)
            print(logits)
        # We need to removed the similarities of samples to themselves
        logits = logits[~torch.eye(2*batch_size, dtype=torch.bool,
                                   device=out0.device)].view(2*batch_size, -1)  # shape: 2*batch_size, 2*batch_size - 1
        if verbose:
            print("logits.shape", logits.shape)
            print(logits)

        # The labels point from a sample in out_i to its equivalent in out_(1-i)
        target = torch.arange(batch_size, device=device, dtype=torch.long)  # shape: batch_size
        target = torch.cat([target + batch_size - 1, target])  # shape: 2*batch_size
        if verbose:
            print("target.shape", target.shape)
            print(target)

        loss = self.cross_entropy(logits, target)  # shape: 2*batch_size before reduction, after reduction is a scalar
        return loss


class Projection(torch.nn.Module):

    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim, bias=False),
            torch.nn.BatchNorm1d(self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        )

    def forward(self, x):
        return self.head(x)


class SimClrModel(LightningModule):
    """
    See
    https://pytorch-lightning.readthedocs.io/en/stable/starter/style_guide.html  and
    https://github.com/PyTorchLightning/Lightning-Bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py#L61-L301
    """

    def __init__(
            self,
            # architecture
            backbone_type: str = 'resnet18',
            number_of_img_channels: int = 9,
            number_backbone_features: int = 256,
            projection_out_dim: int = 128,
            projection_hidden_dim: int = 128,
            # optimizer and scheduler
            checkpoint_interval_min: int = 20,
            optimizer_type: str = 'adam',  # can be 'adam' or 'SGD'
            is_scheduled: bool = False,
            warmup_epochs: int = 10,
            max_epochs: int = 100,
            max_time_min: int = 90,
            start_lr: float = 0.,
            learning_rate: float = 1e-3,
            final_lr: float = 0.,
            weight_decay: float = 1e-6,
            exclude_bn_bias: bool = True,
            check_val_every_n_epoch: int = 10,
            # modules and callables
            train_transform: Optional[Tuple[Callable, Callable]] = None,
            predict_transform: Optional[Dict[str, Callable]] = None,
            compute_moran: Optional[Callable] = None,
            ):
        """
        TODO: write arg explanation
        """
        super().__init__()

        # Next two lines will make checkpointing much simpler
        self.save_hyperparameters()  # all hyperparameters are saved to the checkpoint
        self.neptune_experiment_id = None  # if from scratch neptune_experiment_is is None
        self.check_val_every_n_epoch = check_val_every_n_epoch

        # architecture
        self.backbone_type = backbone_type
        self.number_of_img_channels = number_of_img_channels
        self.number_backbone_features = number_backbone_features
        self.projection_out_dim = projection_out_dim
        self.projection_hidden_dim = projection_hidden_dim

        # optimizer and scheduler
        self.optimizer_type = optimizer_type
        self.is_scheduled = is_scheduled
        self.warmup_epochs = warmup_epochs
        self.max_time_min = max_time_min
        self.max_epochs = max_epochs
        self.start_lr = start_lr
        self.learning_rate = learning_rate
        self.final_lr = final_lr
        self.weight_decay = weight_decay
        self.exclude_bn_bias = exclude_bn_bias
        self.checkpoint_interval_min = checkpoint_interval_min

        self.backbone = self.init_backbone(number_of_img_channels=self.number_of_img_channels,
                                           number_backbone_features=self.number_backbone_features,
                                           backbone_type=self.backbone_type)

        self.projection = Projection(input_dim=self.number_backbone_features,
                                     hidden_dim=self.projection_hidden_dim,
                                     output_dim=self.projection_out_dim)

        self.nt_xent_loss = NTXentLoss()

        # define the transforms
        self.train_transform = train_transform
        self.predict_transform = predict_transform
        self.compute_moran = compute_moran

    @staticmethod
    def init_backbone(number_of_img_channels: int,
                      number_backbone_features: int,
                      backbone_type: str) -> torch.nn.Module:
        if backbone_type == 'resnet18':
            net = torchvision.models.resnet18(pretrained=True)
        elif backbone_type == 'resnet34':
            net = torchvision.models.resnet34(pretrained=True)
        elif backbone_type == 'resnet50':
            net = torchvision.models.resnet50(pretrained=True)
        else:
            raise Exception("backbone_type not recognized. Received ->", backbone_type)

        last_conv_channels = list(net.children())[-1].in_features
        first_conv_out_channels = list(net.children())[0].out_channels
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                number_of_img_channels,
                first_conv_out_channels,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            ),
            *list(net.children())[1:-1],
            torch.nn.Conv2d(last_conv_channels, number_backbone_features, kernel_size=(1, 1)),
        )

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection(y)
        return z, y

    def training_step(self, batch, batch_idx) -> dict:
        list_imgs, list_labels, list_metadata = batch

        # this is data augmentation
        with torch.no_grad():
            img1 = self.train_transform[0](list_imgs)
            img2 = self.train_transform[1](list_imgs)

        z1, y1 = self(img1)
        z2, y2 = self(img2)
        return {'z1': z1, 'z2': z2}

    def training_step_end(self, outputs: List[dict]):
        if isinstance(outputs, list) and isinstance(outputs[0], dict):
            z1 = torch.cat([output['z1'] for output in outputs], dim=0)
            z2 = torch.cat([output['z2'] for output in outputs], dim=0)
        elif isinstance(outputs, dict):
            z1 = outputs['z1']
            z2 = outputs['z2']
        else:
            raise Exception("ERROR. Expected dict or list[dict]. Received {0}".format(type(outputs)))

        loss = self.nt_xent_loss(z1, z2)
        self.log('train_loss', loss, logger=True, on_step=False, on_epoch=True, sync_dist=False)
        self.log('batch_size', z1.shape[0], logger=True, on_step=False, on_epoch=True, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        list_metadata: List[MetadataCropperDataset]
        list_imgs, list_labels, list_metadata = batch

        mydict = {"metadata": list_metadata}
        for name, transf in self.predict_transform.items():
            img = transf(list_imgs)
            z, y = self(img)
            mydict["features_bbone_"+str(name)] = y.flatten(start_dim=1)

        return mydict

    def validation_epoch_end(self, list_of_dict) -> None:
        total_dict = concatenate_list_of_dict(list_of_dict)

        morans_all = numpy.array([metadata.moran for metadata in total_dict["metadata"]])
        fnames_all = numpy.array([metadata.f_name for metadata in total_dict["metadata"]])

        for feature_key in total_dict.keys():
            if feature_key.startswith("feature"):
                features_all = total_dict[feature_key].cpu().numpy()

                try:
                    # regression
                    x_train, x_test, y_train, y_test = train_test_split(
                        features_all,
                        morans_all,
                        stratify=fnames_all,
                        test_size=0.2,
                        random_state=1)

                    mlp_regressor = MLPRegressor(
                        hidden_layer_sizes=[],
                        solver='adam',
                        alpha=0.0001,
                        batch_size='auto',
                        learning_rate='constant',
                        learning_rate_init=0.001,
                        max_iter=100000,
                        shuffle=True,
                        random_state=1,
                        tol=1E-6,
                        verbose=False,
                        n_iter_no_change=100,
                        early_stopping=False)

                    mlp_regressor.fit(x_train, y_train)
                    r2_test = mlp_regressor.score(x_test, y_test)
                    r2_train = mlp_regressor.score(x_train, y_train)
                    self.log("mlp/"+str(feature_key)+"/r2_train", r2_train, logger=True)
                    self.log("mlp/"+str(feature_key)+"/r2_test", r2_test, logger=True)

                    # Classification
                    x_train, x_test, y_train, y_test = train_test_split(
                        features_all,
                        fnames_all,
                        stratify=fnames_all,
                        test_size=0.2,
                        random_state=1)

                    mlp_classification = MLPClassifier(
                        hidden_layer_sizes=[],
                        solver='adam',
                        alpha=0.0001,
                        batch_size='auto',
                        learning_rate='constant',
                        learning_rate_init=0.001,
                        max_iter=100000,
                        shuffle=True,
                        random_state=1,
                        tol=1E-6,
                        verbose=False,
                        n_iter_no_change=100,
                        early_stopping=False)

                    mlp_classification.fit(x_train, y_train)
                    accuracy_test = mlp_classification.score(x_test, y_test)
                    accuracy_train = mlp_classification.score(x_train, y_train)
                    self.log("mlp/"+str(feature_key)+"/accuracy_train", accuracy_train, logger=True)
                    self.log("mlp/"+str(feature_key)+"/accuracy_test", accuracy_test, logger=True)
                except ValueError:
                    self.log("mlp/"+str(feature_key)+"/r2_train", -9.9, logger=True)
                    self.log("mlp/"+str(feature_key)+"/r2_test", -9.9, logger=True)
                    self.log("mlp/"+str(feature_key)+"/accuracy_train", -9.9, logger=True)
                    self.log("mlp/"+str(feature_key)+"/accuracy_test", -9.9, logger=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None) -> dict:

        # https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#predict-step
        ### SEE https://github.com/PyTorchLightning/pytorch-lightning/issues/9380
        STOP

        list_imgs: List[torch.sparse.Tensor]
        list_labels: List[int]
        list_metadata: List[MetadataCropperDataset]
        list_imgs, list_labels, list_metadata = batch

        # Recompute the moran score and update metadata only if necessary
        list_morans = [metadata.moran for metadata in list_metadata]
        list_of_none = [None] * len(list_morans)
        if self.compute_moran is not None and (list_morans == list_of_none):
            list_morans = [self.compute_moran(img).max().item() for img in list_imgs]
            metadata_list = [metadata._replace(moran=moran) for metadata, moran in zip(list_metadata, list_morans)]
            mydict = {"metadata": metadata_list}
        else:
            mydict = {"metadata": list_metadata}

        # Compute the embeddings
        for name, transf in self.predict_transform.items():
            img = transf(list_imgs)
            z, y = self(img)
            mydict["features_bbone_" + str(name)] = y.flatten(start_dim=1)
            mydict["features_head_" + str(name)] = z.flatten(start_dim=1)

        return mydict

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    @staticmethod
    def exclude_from_wt_decay(named_params, weight_decay, skip_list=('bias', 'bn')):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [{
            'params': params,
            'weight_decay': weight_decay
        }, {
            'params': excluded_params,
            'weight_decay': 0.,
        }]

    def configure_optimizers(self):

        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(named_params=self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        if self.optimizer_type == 'SDG':
            optimizer = torch.optim.SGD(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise Exception("ERROR: Optimizer_type is not valid", self.optimizer_type)

        if self.is_scheduled:
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    linear_warmup_and_decay(self.warmup_epochs, self.max_epochs, cosine_decay=True),
                ),
                "interval": "epoch",
                "frequency": 1,
                "srict": False
            }
            return [optimizer], [scheduler]
        else:
            # No scheduler
            return [optimizer]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """ Loading and resuming is handled automatically. Here I am dealing only with the special variables """
        self.neptune_experiment_id = checkpoint.get("neptune_experiment_id", None)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """ Loading and resuming is handled automatically. Here I am dealing only with the special variables """
        checkpoint["neptune_experiment_id"] = getattr(self.trainer.logger, "experiment_id", None)

    def configure_callbacks(self) -> List[Callback]:
        """
        Model specific callbacks. For general info see here:
        https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/core/lightning.html#LightningModule.configure_callbacks

        These two callbacks simply create a ckpt file every XXX minutes and at the end of the training.
        The location and name of the ckpt file can be modified easily.
        """

        ckpt_dir = "saved_ckpt"
        ckpt_name = 'my_checkpoint-{epoch}'  # note the extension .ckpt will be added automatically

        # save one ckpt at the end of training
        ckpt_end = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=ckpt_name,
            save_weights_only=False,
            save_on_train_epoch_end=True,
            save_last=False,
            every_n_epochs=self.max_epochs,
        )

        # save ckpts every XXX minutes
        ckpt_train = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=ckpt_name,
            save_weights_only=False,
            save_on_train_epoch_end=True,
            save_last=False,
            # the following 2 are mutually exclusive. Determine how frequently to save
            train_time_interval=timedelta(minutes=self.checkpoint_interval_min),
            # every_n_epochs=20,
            # The following 3 determine how many ckpt to save simultaneously
            monitor=None,  # 'train_loss',  # can be a metric or None
            save_top_k=1,  # can be only 1 if monitor is None
            mode='min',  # can be only 'min' or 'max'
        )

        return [ckpt_train, ckpt_end]
