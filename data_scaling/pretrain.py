import torch
from torch import optim
from typing import Union
import lightning.pytorch as pl


class Classifier(pl.LightningModule):
    '''
    Classification model used for pretraining before data valuation.

    Args:
      model: torch.nn.Module to perform classification.
      lr: initial learning rate for cosine schedule.
      min_lr: minimum learning rate for cosine schedule.
      weight_decay: L2 regularization.
      loss: loss function for training, e.g., CrossEntropyLoss().
    '''

    def __init__(self,
                 model: torch.nn.Module,
                 lr: float = 1e-3,
                 min_lr: Union[float, None] = None,
                 weight_decay: float = 0.0,
                 loss: torch.nn.Module = torch.nn.CrossEntropyLoss()):
        # Store modules.
        super().__init__()
        self.model = model
        self.loss = loss

        # Set optimization hyperparameters.
        self.lr = lr
        self.min_lr = lr if min_lr is None else min_lr
        self.weight_decay = weight_decay

        # Save hyperparameters.
        self.save_hyperparameters(ignore=['model', 'loss'])

    def forward(self, batch: Union[torch.Tensor, tuple[torch.Tensor]]):
        # Generate predictions.
        x = batch if isinstance(batch, torch.Tensor) else batch[0]
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        # Calculate loss.
        x, y = batch
        pred = self(x)
        return self.loss(pred, y)

    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        # Log loss.
        x, y = batch
        pred = self(x)
        self.log('val_loss', self.loss(pred, y), on_epoch=True, prog_bar=True)
        self.log('val_acc', (pred.argmax(dim=1) == y).float().mean(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        # Log loss.
        x, y = batch
        pred = self(x)
        self.log('test_loss', self.loss(pred, y), on_epoch=True, prog_bar=True)
        self.log('test_acc', (pred.argmax(dim=1) == y).float().mean(), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs, eta_min=self.min_lr, verbose=True)
        return {
            'optimizer': opt,
            'lr_scheduler': scheduler
        }
