from torchsig.datasets.torchsig_narrowband import TorchSigNarrowband
import torchsig.transforms as ST
import random
import numpy as np
from torchsig.transforms import SignalTransform, Transform
from sigclr.modulation_classes import SIGCLR_CLASSES
import pickle
from typing import Tuple, Any
from torchsig.utils.types import SignalData, ModulatedRFMetadata, Signal
import copy


class SigCLRTorchSigNarrowband(TorchSigNarrowband):
    _idx_to_name_dict = dict(zip(range(len(SIGCLR_CLASSES)), SIGCLR_CLASSES))
    _name_to_idx_dict = dict(zip(SIGCLR_CLASSES, range(len(SIGCLR_CLASSES))))

    @staticmethod
    def convert_idx_to_name(idx: int) -> str:
        return SigCLRTorchSigNarrowband._idx_to_name_dict.get(idx, "unknown")

    @staticmethod
    def convert_name_to_idx(name: str) -> int:
        return SigCLRTorchSigNarrowband._name_to_idx_dict.get(name, -1)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Any]:
        #! We only override this class to change the return type
        encoded_idx = pickle.dumps(idx)
        with self.env.begin(db=self.data_db) as data_txn:
            iq_data = pickle.loads(data_txn.get(encoded_idx))

        with self.env.begin(db=self.label_db) as label_txn:
            mod, snr = pickle.loads(label_txn.get(encoded_idx))

        mod = int(mod)
        signal_meta = ModulatedRFMetadata(
            sample_rate=0.0,
            num_samples=iq_data.shape[0],
            complex=True,
            lower_freq=-0.25,
            upper_freq=0.25,
            center_freq=0.0,
            bandwidth=0.5,
            start=0.0,
            stop=1.0,
            duration=1.0,
            bits_per_symbol=0.0,
            samples_per_symbol=0.0,
            excess_bandwidth=0.0,
            class_name=self._idx_to_name_dict[mod],
            class_index=mod,
            snr=snr,
        )
        signal_data: SignalData = SignalData(samples=iq_data)
        signal = Signal(data=signal_data, metadata=[signal_meta])
        if self.use_signal_data:
            signal = self.T(signal)  # type: ignore
            target = self.TT(signal["metadata"])  # type: ignore
            # because we're using our contrastive transforms with the output of this,
            # we return the full signal
            return signal, target
            # return signal["data"]["samples"], target

        signal = self.T(signal)  # type: ignore
        target = (self.TT(mod), snr)  # type: ignore

        # because we're using our contrastive transforms with the output of this, we
        # return the full signal
        return signal, target 
        # return signal["data"]["samples"], target

class SigCLRNarrowbandDataset:
    def __init__(self, root: str, train: bool, impaired: bool,  transforms: list[SignalTransform], target_transform: Transform, use_signal_data: bool):
        self.transforms=transforms
        self.n_views=2
        self.dataset = SigCLRTorchSigNarrowband(root=root,train=train, impaired=impaired, transform=None, target_transform=target_transform, use_signal_data=use_signal_data)


    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        sampled_transforms = random.sample(self.transforms, self.n_views)
        
        x1 = copy.deepcopy(x)
        x2 = copy.deepcopy(x)
        del x

        x1=ST.Compose([sampled_transforms[0],ST.ComplexTo2D()])(x1)['data']['samples']
        x2=ST.Compose([sampled_transforms[1],ST.ComplexTo2D()])(x2)['data']['samples']

        return (x1.astype(np.float32), x2.astype(np.float32)), y

    def __len__(self) -> int:
        return len(self.dataset)

