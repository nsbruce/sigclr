from torchsig.datasets.sig53 import Sig53
import torchsig.transforms as ST
import os
from dataset import SigCLRDataset

contrast_transforms = [
    ST.TimeVaryingNoise(),
    ST.RandomPhaseShift(),
    ST.TimeReversal(),
    ST.RandomTimeShift(),
    # ST.TimeCrop(),
    ST.GainDrift(),
    ST.LocalOscillatorDrift(),
    ST.Clip(),
    ST.SpectralInversion(),
]

runID=os.getenv("RUNID","medsig53")
root_train = os.getenv("ROOT_TRAIN")#,"/project/def-msteve/torchsig/sig53/")

# Specify Sig53 Options
train = True
impaired = True
class_list = list(Sig53._idx_to_name_dict.values())

target_transform = ST.DescToClassIndex(class_list=class_list)

# Instantiate the Sig53 Training Dataset
sig53_train = SigCLRDataset(Sig53(
    root=root_train, 
    train=train, 
    impaired=impaired,
    transform=None,
    target_transform=target_transform,
    use_signal_data=True,
), transforms=contrast_transforms)
print(f'Our training data comes from {root_train}, and has {len(sig53_train)} impaired signals')