from pytorch_lightning import LightningModule
from torch import optim
import torch.nn as nn
import torch
from sigclr.encoders import ResNet50Encoder


class SigCLR(LightningModule):
    def __init__(self, lr: float, temperature: float, weight_decay: float):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"

        self.encoder = ResNet50Encoder(in_chans=2, pretrained=False)

        print("Device type:", self.device)

        self.encoder.to(self.device)

        self.temperature = temperature
        self.similarity = nn.CosineSimilarity(dim=1)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        # the output of the resnet 50 encoder is the embedding space of size 2048. We 
        # reduce this in the projection head to a smaller latent space of 128 for the
        # contrastive loss computation. 2048 and 128 are typical numbers.
        self.projection_head=nn.Sequential(
            nn.Linear(self.encoder.backbone.num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, xi, xj):
        hi, hj = self.encoder(xi), self.encoder(xj)
        zi, zj = self.projection_head(hi), self.projection_head(hj) 
        return zi, zj, hi, hj

    def predict(self, x):
        with torch.no_grad():
            h = self.encoder(x)
            z = self.projection_head(h)
        return z, h

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def normalized_temp_scaled_cross_entropy_loss(self, zi, zj) -> float:
        # z = torch.cat((zi, zj), dim=0)
        sim = self.similarity(zi, zj) / self.temperature
        return sim

    def training_step(self, batch, batch_idx):
        # batch shape is list of length 2. First element is two tensors (one for each)
        # input signal with torch.size([batch_size, 2, 512]). The second element is the
        # class labels with torch.size([batch_size])
        (xi, xj), _ = batch
        zi, zj, hi, hj = self.forward(xi, xj)
        loss = self.normalized_temp_scaled_cross_entropy_loss(zi, zj)
        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (xi, xj), _ = batch
        zi, zj, hi, hj = self.forward(xi, xj)
        loss = self.normalized_temp_scaled_cross_entropy_loss(zi, zj)
        # self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
