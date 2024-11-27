from torchsig.datasets import conf
from dataclasses import dataclass
from torchsig.utils.writer import DatasetCreator, DatasetLoader
from torchsig.datasets.modulations import ModulationsDataset
from torchsig.utils.dataset import collate_fn
import os
import tqdm
import numpy as np
import click
from sigclr.modulation_classes import SIGCLR_CLASSES

@dataclass
class SigCLRNarrowbandCleanTrainConfig(conf.NarrowbandCleanTrainConfig):
    num_samples = len(SIGCLR_CLASSES) * 500_000
    num_iq_samples = 512

@dataclass
class SigCLRNarrowbandCleanValConfig(conf.NarrowbandCleanValConfig):
    num_samples = len(SIGCLR_CLASSES) * 12_500
    num_iq_samples = 512

batch_size = 4096  #TODO decide?
num_workers = os.cpu_count() // 2


class SigCLRDatasetCreator(DatasetCreator):
    """A subclass of the Torchsig DatasetCreator in order to reduce data volume"""
    def create(self):
        if self.writer.exists():
            print("Dataset already exists in {}. Not regenerating".format(self.path))
            return

        for batch in tqdm.tqdm(self.loader, total=len(self.loader)):
            # This reduces the data size from np.complex128 to np.complex64
            data, label = batch
            data = tuple(x.astype(np.complex64) for x in data)
            self.writer.write((data, label))

@click.command
@click.option("--path", default="narrowband", help="Path to generate narrowband datasets")
def generate(path: str) -> None:
    for config in [SigCLRNarrowbandCleanValConfig, SigCLRNarrowbandCleanTrainConfig]:
        ds = ModulationsDataset(
            classes=SIGCLR_CLASSES,
            level=config.level,
            num_samples=config.num_samples,
            num_iq_samples=config.num_iq_samples,
            use_class_idx=config.use_class_idx,
            include_snr=config.include_snr,
            eb_no=config.eb_no,
        )
        dataset_loader = DatasetLoader(ds, seed=12345678, collate_fn=collate_fn, num_workers=num_workers, batch_size=batch_size)
        creator = SigCLRDatasetCreator(ds, seed=12345678, path="{}".format(os.path.join(path, config.name)), loader=dataset_loader, num_workers=num_workers)
        creator.create()
    

if __name__ == '__main__':
    generate()