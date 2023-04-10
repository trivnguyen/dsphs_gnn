
import torch
import pytorch_lightning as pl

from . import gnn, transforms
from .logger import logger

class DataModule(pl.LightningModule):

    def __init__(
        self, model_hparams, graph_hparams, optimizer_hparams,
        num_posteriors=5000):
        super().__init__()
        self.save_hyperparameters()
        logger.info("Run Hyperparameters:")
        for hparams in self.hparams:
            logger.info(f"{hparams}: {self.hparams[hparams]}")
        self.model = gnn.GNNRegressor(**model_hparams)
        self.graph_transforms = transforms.GraphTransforms(**graph_hparams)
        self.num_posteriors = num_posteriors

    def forward(self, x, *args, **kargs):
        return self.model(x, *args, **kargs)

    def configure_optimizers(self):
        """ Initialize optimizer and LR scheduler """
        optimizer = torch.optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 'min', patience=4),
                'monitor': 'train_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def training_step(self, batch, batch_idx):
        x =  self.model(batch.x, batch.edge_index, batch.batch, batch.edge_weight)
        log_prob = self.model.maf.log_prob(batch.y, context=x)
        loss = -log_prob.mean()
        self.log('train_loss', loss, on_epoch=True, batch_size=len(batch.x))
        return loss

    def validation_step(self, batch, batch_idx):
        x =  self.model(batch.x, batch.edge_index, batch.batch, batch.edge_weight)
        log_prob = self.model.maf.log_prob(batch.y, context=x)
        loss = -log_prob.mean()
        self.log('val_loss', loss, on_epoch=True, batch_size=len(batch.x))
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = self.model(batch.x, batch.edge_index, batch.batch, batch.edge_weight)
        y_pred = self.model.maf.sample(num_samples=self.num_posteriors, context=x)
        if batch.y is not None:
            return y_pred, batch.y
        return y_pred


