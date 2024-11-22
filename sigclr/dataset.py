from torchsig.utils.dataset import SignalDataset
from torchsig.datasets.torchsig_narrowband import TorchSigNarrowband
import torchsig.transforms as ST
import random
import numpy as np
from torchsig.transforms import SignalTransform, Transform


class SigCLRDataset(SignalDataset):
    def __init__(self, dataset,transforms=None):
        self.dataset = dataset
        self.transforms=transforms
        self.n_views=2
        if self.transforms is None:
            self.transforms=[ST.Identity(),ST.Identity()]            
        super().__init__(dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        sampled_transforms = random.sample(self.transforms, self.n_views)
        x1=ST.Compose([sampled_transforms[0],ST.ComplexTo2D()])(x)
        x2=ST.Compose([sampled_transforms[1],ST.ComplexTo2D()])(x)

        return (x1.astype(np.float32), x2.astype(np.float32)),y

    def __len__(self) -> int:
        return len(self.dataset)

class SigCLRNarrowbandDataset:
    def __init__(self, root: str, train: bool, impaired: bool,  transforms: list[SignalTransform], target_transform: Transform, use_signal_data: bool):
        self.transforms=transforms
        self.n_views=2
        if self.transforms is None:
            self.transforms=[ST.Identity(),ST.Identity()]            
        self.dataset = TorchSigNarrowband(root=root,train=train, impaired=impaired, transform=None, target_transform=target_transform, use_signal_data=use_signal_data)


    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        sampled_transforms = random.sample(self.transforms, self.n_views)
        x1=ST.Compose([sampled_transforms[0],ST.ComplexTo2D()])(x)
        x2=ST.Compose([sampled_transforms[1],ST.ComplexTo2D()])(x)

        return (x1.astype(np.float32), x2.astype(np.float32)), y

    def __len__(self) -> int:
        return len(self.dataset)