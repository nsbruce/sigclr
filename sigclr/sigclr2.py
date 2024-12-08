from pytorch_lightning import LightningModule
from torch import optim
import torch.nn as nn
import torch
from sigclr.encoders import ResNet50Encoder
import torch.nn.functional as F

class SigCLR(LightningModule):
    def __init__(self, lr: float, temperature: float, weight_decay: float):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"

        self.encoder = ResNet50Encoder(in_chans=2, pretrained=False)

        # self.encoder.to(self.device)

        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        # the output of the resnet 50 encoder is the embedding space of size 2048. We 
        # reduce this in the projection head to a smaller latent space of 128 for the
        # contrastive loss computation. 2048 and 128 are typical numbers.
        self.projection_head=nn.Sequential(
            nn.Linear(self.encoder.backbone.num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, xi: torch.Tensor, xj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # xi and xj have shapes torch.Size([batch_size, 2, signal_length])

        # hi and hj have shapes torch.Size([batch_size, 2048]) where the 2048 comes from the embedding space of the resnet50 encoder
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
    
    def normalized_temp_scaled_cross_entropy_loss(self, zi: torch.Tensor, zj: torch.Tensor) -> torch.Tensor:
        # zi and zj shapes are of torch.size([batch_size, 128])

        # normalize embeddings to encourage a focus on the direction of the embedding not the magnitude
        # output shape from normalize is torch.Size([batch_size, 128]) since no dimensional change
        zi = F.normalize(zi, dim=1)
        zj = F.normalize(zj, dim=1)

        # need to compute the cosine similarity matrix between zi and zj, which is defined of dot product of normalized zi and zj
        # output shape from matmul is torch.Size([batch_size, batch_size])
        sim = torch.matmul(zi, zj.T) / self.temperature

        # the class labels are just indices of the embeddings showing that each pair from zi and zj is similar and disimilar from all other pairs
        # labels shape is torch.Size([batch_size])
        labels = torch.arange(0, zi.size(0), device=self.device)

        # the cross entropy loss is computed between the cosine similarity matrix and the class labels
        # the loss is a torch.Tensor but with a shape of torch.Size([]) It's just a float wrapped in a torch.Tensor to keep it in torch-land
        loss = F.cross_entropy(sim, labels)

        return loss

    def training_step(self, batch: list[tuple[torch.Tensor] | torch.Tensor], batch_idx: int) -> torch.Tensor:

        # batch is a list of length 2. First element is two tensors (one for each)
        # input signal with torch.size([batch_size, 2, 512]). The second element is the
        # class labels with torch.size([batch_size])
        (xi, xj), _ = batch
        zi, zj, hi, hj = self.forward(xi, xj)
        loss = self.normalized_temp_scaled_cross_entropy_loss(zi, zj)
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: list[tuple[torch.Tensor] | torch.Tensor], batch_idx: int) -> torch.Tensor:
        (xi, xj), _ = batch
        zi, zj, hi, hj = self.forward(xi, xj)
        loss = self.normalized_temp_scaled_cross_entropy_loss(zi, zj)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        return loss
