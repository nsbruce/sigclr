# from torchsig.models.iq_models.efficientnet.efficientnet import efficientnet_b4
from torchsig.models import EfficientNet1d
import torch.nn as nn
import torch
import timm
from torchsig.models.model_utils.model_utils_1d.conversions_to_1d import convert_2d_model_to_1d

class EfficientNetB4Encoder(nn.Module):
    def __init__(self, pretrained=True,path="/project/def-msteve/torchsig-pretrained-models/sig53/efficientnet_b4_online.pt",neck_out_features=53, neck_hidden_features=512,dropout_rate=0.2):
        super().__init__()
        # self.backbone = efficientnet_b4(pretrained=pretrained, path=path)
        self.backbone = EfficientNet1d(
            input_channels=2,
            n_features=53,
            efficientnet_version="b4"
        )
        if pretrained:
            self.backbone.load_state_dict(torch.load(path))

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

class ResNet50Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.neck_out_features=53
        self.model = convert_2d_model_to_1d(timm.create_model("resnet50", in_chans=2, num_classes=self.neck_out_features))

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        with torch.no_grad():
            out = self.forward(x)
        return out
