from typing import Callable, Optional
import torch
from torch import nn
import torchmetrics
import pytorch_lightning as pl


class Classifier(pl.LightningModule):
    """
    Supervised Linear Classifier using the CL model output embeddings
    
    """

    def __init__(
        self,
        embedding_out_dim: int,
        out_dim: int,
        loss: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None):
        """
        Args:
            embedding_out_dim: Dimension of the output embedding vector
            out_dim: Number of classes given
            loss: Any callable which takes inputs, i.e. y=Net(x), and target and produce a loss
        """
        super().__init__()
        self.classifier = nn.Linear(embedding_out_dim, out_dim)
        self.loss = torch.nn.CrossEntropyLoss() if loss is None else loss
        self.metrics = {
            "accuracy": torchmetrics.Accuracy(), 
        }

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[:2]
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("train_classifier_loss", loss, on_epoch=True, on_step=False, prog_bar=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]
        y_hat = self.forward(x)
        y_hat = torch.softmax(y_hat, dim=-1)
        
        for metric_name, metric in self.metrics.items():
            metric(y_hat.cpu(), y.cpu())
            self.log(f"val_{metric_name}", metric.compute(), on_epoch=True, on_step=False, prog_bar=False)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1E-3)
        return optim
    