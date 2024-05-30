import torch
import lightning.pytorch as pl
from torch import optim
from typing import Union
from torch.distributions import Normal
from timm.scheduler import CosineLRScheduler


class LikelihoodModel(pl.LightningModule):
    '''
    Amortized scaling law estimator. The model is trained to predict the
    scaling law parameters (c, alpha, sigma, beta) for a given (x, y) pair.

    Args:
      model: torch.nn.Module that predicts scaling parameters for each class.
      lr: initial learning rate for cosine schedule.
      min_lr: minimum learning rate for cosine schedule.
      weight_decay: L2 regularization.
      alpha_penalty: penalty for alpha deviating from 1, helps stabilize training.
      save_architecture: whether to save the model architecture in the checkpoints.
    '''

    def __init__(self,
                 model: torch.nn.Module,
                 lr: float = 1e-3,
                 min_lr: Union[float, None] = None,
                 weight_decay: float = 0.0,
                 alpha_penalty: float = 1e-5,
                 save_architecture: bool = False):
        # Setup.
        super().__init__()
        self.model = model

        # Set optimization hyperparameters.
        self.lr = lr
        self.min_lr = lr if min_lr is None else min_lr
        self.weight_decay = weight_decay
        self.alpha_penalty = alpha_penalty

        # Save hyperparameters.
        if save_architecture:
            self.save_hyperparameters()
        else:
            self.save_hyperparameters(ignore=['model'])

    def forward(self, batch: Union[torch.Tensor, tuple[torch.Tensor]]):
        # Prepare input.
        if isinstance(batch, torch.Tensor):
            batch = (batch,)

        # Predict all scaling law parameters.
        x = batch[0]
        pred = self.model(x)
        c_sign, log_c_abs, log_alpha, log_sigma, log_beta = torch.split(pred, pred.shape[1] // 5, dim=1)
        alpha = torch.exp(log_alpha)
        beta = torch.exp(torch.clamp(log_beta, max=2.0))

        # Decode predictions for specified class.
        if len(batch) != 1:
            y = batch[1]
            if len(y.shape) != 1:
                y = y.argmax(dim=1)
            y = y.unsqueeze(1)
            c_sign = torch.gather(c_sign, 1, y)
            log_c_abs = torch.gather(log_c_abs, 1, y)
            alpha = torch.gather(alpha, 1, y)
            log_sigma = torch.gather(log_sigma, 1, y)
            beta = torch.gather(beta, 1, y)

        # Rescale predictions if not training.
        if not self.training:
            c = torch.sign(c_sign) * torch.exp(log_c_abs)
            sigma = torch.exp(log_sigma)
            return {'c': c, 'alpha': alpha, 'sigma': sigma, 'beta': beta}
        else:
            return {'c_sign': c_sign, 'log_c_abs': log_c_abs, 'alpha': alpha, 'log_sigma': log_sigma, 'beta': beta}

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        # Generate predictions.
        x, y, samples, cardinalities = batch
        pred = self((x, y))
        c_sign = pred['c_sign']
        log_c_abs = pred['log_c_abs']
        alpha = pred['alpha']
        log_sigma = pred['log_sigma']
        beta = pred['beta']

        # Setup for Gaussian NLL loss.
        loc = torch.exp(log_c_abs - log_sigma.detach()) * (cardinalities ** (beta.detach() / 2 - alpha))
        scale = torch.exp(log_sigma - log_sigma.detach()) * (cardinalities ** (beta.detach() / 2 - beta / 2))
        samples = samples / (torch.exp(log_sigma.detach()) * (cardinalities ** (- beta.detach() / 2)) + 1e-12)
        # samples = samples * torch.exp(- log_sigma.detach()) * (cardinalities ** (beta.detach() / 2))

        # Diagnose any numerical stability issues.
        if torch.isnan(loc).any() or torch.isnan(scale).any() or torch.isnan(samples).any():
            print('NaN encountered in loc, scale or samples')
            print(f'loc: {torch.isnan(loc).any()}, '
                  f'scale: {torch.isnan(scale).any()}, '
                  f'samples: {torch.isnan(samples).any()}')
            print(f'alpha min/max: {alpha.min()}, {alpha.max()}'
                  f'beta min/max: {beta.min()}, {beta.max()}'
                  f'diff min/max: {(beta / 2 - alpha).min()}, {(beta / 2 - alpha).max()}')
        elif torch.isinf(loc).any() or torch.isinf(scale).any() or torch.isinf(samples).any():
            print('Inf encountered in loc, scale or samples')
            print(f'loc min/max: {loc.min()}, {loc.max()}'
                  f'scale min/max: {scale.min()}, {scale.max()}'
                  f'samples min/max: {samples.min()}, {samples.max()}')
            print(f'alpha min/max: {alpha.min()}, {alpha.max()}'
                  f'beta min/max: {beta.min()}, {beta.max()}'
                  f'diff min/max: {(beta / 2 - alpha).min()}, {(beta / 2 - alpha).max()}')

        # Calculate Gaussian NLL loss with both signs.
        dist = Normal(loc=loc, scale=scale)
        loss_cat = torch.stack([- dist.log_prob(- samples).mean(dim=1),
                                - dist.log_prob(samples).mean(dim=1)], dim=1)
        argmin = torch.argmin(loss_cat, dim=1)
        gaussian_loss = torch.gather(loss_cat, 1, argmin.unsqueeze(1)).mean()

        # Calculate classification loss for sign.
        sign_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            c_sign.squeeze(), argmin.float())

        # Alpha prior.
        prior = self.alpha_penalty * torch.mean((alpha - 1) ** 2)

        return gaussian_loss + sign_loss + prior

    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        # Generate predictions.
        x, y, samples, cardinalities = batch
        pred = self((x, y))
        c = pred['c']
        alpha = pred['alpha']
        sigma = pred['sigma']
        beta = pred['beta']

        # Calculate loss.
        mean = c * (cardinalities ** (- alpha))
        std = sigma * (cardinalities ** (- beta / 2))
        dist = Normal(loc=mean, scale=std)
        loss = - dist.log_prob(samples).mean()
        self.log('val_loss', loss, prog_bar=True, batch_size=len(x))

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=1e-6)
        scheduler = CosineLRScheduler(opt, t_initial=self.trainer.max_epochs, warmup_lr_init=1e-8,
                                      warmup_t=5, lr_min=self.min_lr)
        return {'optimizer': opt, 'lr_scheduler': scheduler}

    def lr_scheduler_step(self, scheduler, metric):
        # Modified for Timm cosine scheduler.
        scheduler.step(epoch=self.current_epoch)


class NaiveLikelihoodModel(pl.LightningModule):
    '''
    Amortized scaling law estimator. The model is trained to predict the
    scaling law parameters (c, alpha, sigma, beta) for a given (x, y) pair.

    This implementation is naive in two senses:
      1. We predict c directly rather than separately predicting the sign and magnitude
      2. We calculate Gaussian NLL without rescaling to have std \approx 1

    Args:
      model: torch.nn.Module that predicts scaling parameters for each class.
      target_scaling: float to rescale predictions to improve numerical stability.
      lr: initial learning rate for cosine schedule.
      min_lr: minimum learning rate for cosine schedule.
      save_architecture: whether to save the model architecture in the checkpoints.
    '''

    def __init__(self,
                 model: torch.nn.Module,
                 target_scaling: float = 1.0,
                 lr: float = 1e-3,
                 min_lr: float = 1e-6,
                 save_architecture: bool = False):
        # Setup.
        super().__init__()
        self.model = model

        # Set optimization hyperparameters.
        self.target_scaling = target_scaling
        self.lr = lr
        self.min_lr = min_lr

        # Save hyperparameters.
        if save_architecture:
            self.save_hyperparameters()
        else:
            self.save_hyperparameters(ignore=['model'])

    def forward(self, batch: Union[torch.Tensor, tuple[torch.Tensor]]):
        # Prepare input.
        if isinstance(batch, torch.Tensor):
            batch = (batch,)

        # Predict all scaling law parameters.
        x = batch[0]
        pred = self.model(x)
        c, log_alpha, log_sigma, log_beta = torch.split(pred, pred.shape[1] // 4, dim=1)
        alpha = torch.exp(torch.clamp(log_alpha, max=2.0))
        sigma = torch.exp(torch.clamp(log_sigma, max=5.0)) + 1e-3
        beta = torch.exp(torch.clamp(log_beta, max=2.0))

        # Rescale predictions if not training.
        if not self.training:
            c = c * self.target_scaling
            sigma = sigma * self.target_scaling

        # Decode predictions for specified class.
        if len(batch) != 1:
            y = batch[1]
            if len(y.shape) != 1:
                y = y.argmax(dim=1)
            y = y.unsqueeze(1)
            c = torch.gather(c, 1, y)
            alpha = torch.gather(alpha, 1, y)
            sigma = torch.gather(sigma, 1, y)
            beta = torch.gather(beta, 1, y)

        return {'c': c, 'alpha': alpha, 'sigma': sigma, 'beta': beta}

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        # Generate predictions.
        x, y, samples, cardinalities = batch
        pred = self((x, y))
        c = pred['c']
        alpha = pred['alpha']
        sigma = pred['sigma']
        beta = pred['beta']

        # Calculate loss.
        mean = c * (cardinalities ** (- alpha))
        std = sigma * (cardinalities ** (- beta / 2))
        dist = Normal(loc=mean, scale=std)
        loss = - dist.log_prob(samples / self.target_scaling).mean()
        return loss

    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        # Generate predictions.
        x, y, samples, cardinalities = batch
        pred = self((x, y))
        c = pred['c']
        alpha = pred['alpha']
        sigma = pred['sigma']
        beta = pred['beta']

        # Calculate loss.
        mean = c * (cardinalities ** (- alpha))
        std = sigma * (cardinalities ** (- beta / 2))
        dist = Normal(loc=mean, scale=std)
        loss = - dist.log_prob(samples).mean()
        self.log('val_loss', loss, prog_bar=True, batch_size=len(x))

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = CosineLRScheduler(opt, t_initial=self.trainer.max_epochs, warmup_lr_init=1e-8,
                                      warmup_t=5, lr_min=self.min_lr)
        return {'optimizer': opt, 'lr_scheduler': scheduler}

    def lr_scheduler_step(self, scheduler, metric):
        # Modified for Timm cosine scheduler.
        scheduler.step(epoch=self.current_epoch)
