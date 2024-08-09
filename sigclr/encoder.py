from torchsig.models.iq_models.efficientnet.efficientnet import efficientnet_b4
import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, pretrained=True,path="/project/def-msteve/torchsig-pretrained-models/sig53/efficientnet_b4_online.pt",neck_out_features=53, neck_hidden_features=512,dropout_rate=0.2):
        super().__init__()
        self.backbone = efficientnet_b4(pretrained=pretrained, path=path)
        self.neck_out_features=neck_out_features
        self.neck_hidden_features=neck_hidden_features
        self.clsf_in_features=self.backbone.classifier.in_features 
        self.clsf_out_features=self.backbone.classifier.out_features
        self.backbone.classifier = nn.Identity()
        # freeze the backbone
        self.backbone.requires_grad_(False)       
        self.backbone.eval()

        self.neck = nn.Sequential(
            nn.Linear(self.clsf_in_features, self.neck_hidden_features),
            nn.BatchNorm1d(self.neck_hidden_features),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.neck_hidden_features, self.neck_out_features)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            out = self.forward(x)
        return out
