from pytorch_lightning import LightningModule
from torch import optim
import torch.nn as nn
import torch
from sigclr.encoder import Encoder


class BatchSync(torch.autograd.Function):
    """Gathers and Syncs all minibatches from all GPUs into an effective batch."""
    @staticmethod
    def forward(ctx, minibatch): #minibatch on each GPU/process 
        ctx.batch_shape = minibatch.shape
        batch = [torch.zeros(ctx.batch_shape) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(batch, minibatch)
        return tuple(batch)

    @staticmethod
    def backward(ctx, *grads):
        grad_out = torch.zeros(ctx.batch_shape)
        grad_out[:] = grads[torch.distributed.get_rank()]
        return grad_out

class SigCLR(LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, batch_size, max_epochs, device, freeze_backbone):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"

        self.encoder = Encoder()
        if freeze_backbone:
            # freeze the pretrained convnet
            self.encoder.backbone.eval()
            self.encoder.backbone.requires_grad = False

        # send to device
        self.encoder.to(device)

        self.temperature = temperature
        self.batch_size=batch_size
        self.hparams.device=device
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        self.projection_head=nn.Sequential(
            nn.Linear(self.encoder.neck_out_features,hidden_dim),
            nn.BatchNorm1d(hidden_dim), #BM: we might this to speed up our training
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, self.encoder.neck_out_features, bias=False)
        )
        self.world_size=1
        if torch.distributed.is_initialized():
            self.world_size=torch.distributed.get_world_size()

    def forward(self, xi, xj):
        hi, hj = self.encoder(xi), self.encoder(xj)
        zi, zj = self.projection_head(hi), self.projection_head(hj) 
        return zi, zj, hi, hj

    def predict(self, x):
        with torch.no_grad():
            # zi, zj, hi, hj = self.forward(x)
            h = self.encoder(x)
            z = self.projection_head(h)
        return z, h

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            # T_max=self.hparams.max_epochs,
            T_max=3,
            eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def ntXent_loss(self, batch, mode="train"):
        (xi,xj), _ = batch
        #if xi.shape[0]!=self.batch_size: # Recompute the mask
        self.batch_size=xi.shape[0]
        self.N=self.batch_size*self.world_size 
        self.allN=2*self.N #The batch is effectively 2*batch_size*number_of_GPUs batchs from the GPUs/processes
        
        self.mask = torch.ones((self.allN, self.allN), dtype=bool,device=self.device)
        self.mask = self.mask.fill_diagonal_(0)
        for i in range(self.N):
            self.mask[i, self.N + i] = 0
            self.mask[self.N + i, i] = 0
            
        zi,zj, hi, hj = self.forward(xi,xj)
	
	# Gather and sync the rest of minibatches results:
        if self.world_size > 1:
            z_i = torch.cat(BatchSync.apply(z_i), dim=0)
            z_j = torch.cat(BatchSync.apply(z_j), dim=0)
        z = torch.cat((zi, zj), dim=0)
        
        sim = self.similarity(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.N)
        sim_j_i = torch.diag(sim, -self.N)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(self.allN, 1)
        negative_samples = sim[self.mask].reshape(self.allN, -1)

        labels = torch.zeros(self.allN).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= self.allN

        # Logging loss
        self.log(mode + "_loss", loss, sync_dist=True, on_step=True, on_epoch=True,prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.ntXent_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.ntXent_loss(batch, mode="val")
