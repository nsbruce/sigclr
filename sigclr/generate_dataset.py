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
from time import time


@dataclass
class SigCLRNarrowbandCleanTrainConfig(conf.NarrowbandCleanTrainConfig):
    num_samples = len(SIGCLR_CLASSES) * 500_000
    num_iq_samples = 256

@dataclass
class SigCLRNarrowbandCleanValConfig(conf.NarrowbandCleanValConfig):
    num_samples = len(SIGCLR_CLASSES) * 12_500
    num_iq_samples = 256

@dataclass
class SigCLRNarrowbandCleanQAConfig(conf.NarrowbandCleanTrainQAConfig):
    seed: int = 1234567893
    num_samples: int = len(SIGCLR_CLASSES) * 2
    num_iq_samples = 256

num_workers = os.cpu_count() - 1


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
@click.option("--batch-size", type=int, default=4096, help="Batch size for dataset")
@click.option("--num-iq-samples", type=int, help="Number of IQ samples per signal")
@click.option("--train", is_flag=True, default=False, help="Whether to generate the training set")
@click.option("--val", is_flag=True, default=False, help="Whether to generate the val set")
@click.option("--qa", is_flag=True, default=False, help="Whether to generate the QA set")
def generate(path: str, batch_size: int, num_iq_samples: int, train: bool, val: bool, qa: bool) -> None:

    configs = []
    if train:
        configs.append(SigCLRNarrowbandCleanTrainConfig)
    if val:
        configs.append(SigCLRNarrowbandCleanValConfig)
    if qa:
        configs.append(SigCLRNarrowbandCleanQAConfig)

    print('Num IQ samples:', num_iq_samples, '(if None means using config default)')
    print('Batch size:', batch_size)    
    print("Datasets to generate:", [c.name for c in configs])

    startt = time()
    for config in [SigCLRNarrowbandCleanValConfig, SigCLRNarrowbandCleanTrainConfig]:  #,SigCLRNarrowbandCleanQAConfig]:#, 
        output_path = "{}".format(os.path.join(path, config.name))

        print("Building", output_path)

        ds = ModulationsDataset(
            classes=SIGCLR_CLASSES,
            level=config.level,
            num_samples=config.num_samples,
            num_iq_samples=config.num_iq_samples if num_iq_samples is None else num_iq_samples,
            use_class_idx=config.use_class_idx,
            include_snr=config.include_snr,
            eb_no=config.eb_no,
        )

        dataset_loader = DatasetLoader(ds, seed=1234567893, collate_fn=collate_fn, num_workers=num_workers, batch_size=batch_size)
        creator = SigCLRDatasetCreator(ds, seed=1234567893, path=output_path, loader=dataset_loader, num_workers=num_workers)
        creator.create()

    print("Done in", time()-startt, 'seconds')


if __name__ == '__main__':
    generate()
