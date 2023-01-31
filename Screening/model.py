from pickletools import optimize

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from resnet import resnet10, resnet18, resnet50, resnet200


class LitModel(pl.LightningModule):
  def __init__(self, num_layers=50):
    super().__init__()
    if num_layers == 50:
      model = resnet50
    elif num_layers == 10:
      model = resnet10
    elif num_layers == 18:
      model = resnet18
    elif num_layers == 200:
      model = resnet200
    self.model = model(num_classes=1,
                       sample_size=32,
                       sample_duration=32,
                       num_channels=1)
  def forward(self, x):
    return self.model(x)
  
  def training_step(self, batch, batch_idx):
    grid, y = batch
    y_hat = self(grid)
    loss = F.mse_loss(y_hat, y.unsqueeze(1).float())
    self.log("train/loss", loss, prog_bar=True)
    return loss
  
  def validation_step(self, val_batch, batch_idx):
    grid, y = val_batch
    y_hat = self(grid)
    torch.save(y, f"val_real_batch_{batch_idx}.pt")
    torch.save(y_hat, f"val_pred_batch_{batch_idx}.pt")
    loss = F.mse_loss(y_hat, y.unsqueeze(1).float())

    self.log('val_loss', loss)
    self.log("validation/loss", loss, prog_bar=True)
    return loss

  def test_step(self, test_batch, batch_idx):
    grid, y = test_batch
    y_hat = self(grid)
    loss = F.mse_loss(y_hat, y.unsqueeze(1).float())
    torch.save(y, f"test_real_batch_{batch_idx}.pt")
    torch.save(y_hat, f"test_pred_batch_{batch_idx}.pt")
    self.log("test_loss/loss", loss)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    return [optimizer], [lr_scheduler]
    # return optimizer
