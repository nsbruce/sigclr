from torchsig.models.iq_models.efficientnet.efficientnet import efficientnet_b4
import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, pretrained=True,path="/project/def-msteve/torchsig-pretrained-models/sig53/efficientnet_b4_online.pt",device='cuda'):
        super().__init__()
        self.convnet = efficientnet_b4(pretrained=pretrained, path=path)

        self.clsf_in_features=self.convnet.classifier.in_features 
        self.clsf_out_features=self.convnet.classifier.out_features
        self.convnet.classifier = None

        self.convnet = self.convnet.to(device)
    def forward(self, x):
        return self.convnet(x)

    def predict(self, x):
        with torch.no_grad():
            out = self.forward(x)
        return out
