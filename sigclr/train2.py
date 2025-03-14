from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader
import torchsig.transforms as ST
import torch
import os
import click
from sigclr.dataset2 import SigCLRNarrowbandDataset
from sigclr.sigclr2 import SigCLR
from sigclr.modulation_classes import SIGCLR_CLASSES
from lightning.fabric import Fabric

contrast_transforms = [
    ST.TimeVaryingNoise(),
    ST.RandomPhaseShift(),
    ST.TimeReversal(),
    ST.RandomTimeShift(),
    ST.GainDrift(),
    ST.LocalOscillatorDrift(),
    ST.Clip(),
    ST.SpectralInversion(),
]


def setup_datasets(impaired: bool, batch_size: int):
    root_train = os.getenv("ROOT_TRAIN")
    root_val = os.getenv("ROOT_VAL")

    torch.set_float32_matmul_precision('medium')
    # num_workers = os.cpu_count()-1
    num_workers = os.cpu_count()//4
    torch.backends.cudnn.deterministic = True

    print(f"Number of workers: {num_workers}")


    target_transform = ST.DescToClassIndex(class_list=SIGCLR_CLASSES)

    # Instantiate the training dataset
    train_dataset = SigCLRNarrowbandDataset(
        root=root_train, 
        train=True, 
        impaired=impaired,
        target_transform=target_transform,
        use_signal_data=True,
        transforms=contrast_transforms
    )
    print(f'Training data comes from {root_train}, and has {len(train_dataset)} signals')

    # Instantiate the validation dataset
    val_dataset = SigCLRNarrowbandDataset(
        root=root_val, 
        train=False, 
        impaired=impaired,
        target_transform=target_transform,
        use_signal_data=True,
        transforms=contrast_transforms
    )
    print(f'Validation data comes from {root_val}, and has {len(val_dataset)} signals')

    train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=num_workers,
        )
    val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=num_workers,
        )

    return train_loader, val_loader


def train_internal(batch_size, epochs, checkpoint_file, optimizer):
    lr=0.001  # for optimizer
    weight_decay=1e-4  # for optimizer
    temperature=0.07  # for ntXent loss computation
    
    train_loader, val_loader = setup_datasets(impaired=False, batch_size=batch_size)
    checkpoint_path=os.getenv("CHECKPOINT_PATH")

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, every_n_epochs=1, mode="min", monitor="val_loss", save_top_k=3,save_last=True)

    trainer = Trainer(
        default_root_dir=checkpoint_path,
        devices="auto",
        accelerator="auto",
        num_nodes=int(os.environ['SLURM_JOB_NUM_NODES']),
        max_epochs=epochs,
        enable_progress_bar=False,
        callbacks=checkpoint_callback,
        strategy="ddp",  # from pytorch_lightning docs about running on slurm
        # accumulate_grad_batches=2,  # simulates larger batch size somehow
        precision="16-mixed",
        # sync_batchnorm=True,  # not sure if I want this or not
    )
    print("Trainer accelerator: ", trainer.accelerator.__class__.__name__)
    print("Trainer num. devices:", trainer.num_devices)

    # If a pretrained model was passed, load it and train some more.
    if checkpoint_file is not None and os.path.isfile(checkpoint_file):
        print(f"Was provided weights at {checkpoint_file}.")
        # Load the model with the saved hyperparameters
        model = SigCLR.load_from_checkpoint(checkpoint_file)

        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_file)

    # if a pretrained model was passed which doesn't exist, raise an error
    elif checkpoint_file is not None and not os.path.isfile(checkpoint_file):
        raise RuntimeError(f"A checkpoint file was passed, but was not found ({checkpoint_file})")

    # if no pretrained model was passed, build a new model
    else:
        print("No checkpoint passed. Instantiating a new model.")
        seed_everything(42)  # To be reproducable
        model = SigCLR(lr=lr, temperature=temperature, weight_decay=weight_decay, optimizer_name=optimizer)
        trainer.fit(model, train_loader, val_loader)

    return model

@click.command()
@click.option('--batch-size', type=int, help='Batch size used during training and validation.')
@click.option('--epochs', type=int, help='Number of epochs during training.')
@click.option('--checkpoint-file', help='Restarts from the provided previous checkpointed model file.')
@click.option('--optimizer', type=str, help="Must be one of 'AdamW' or 'LARS'")
def train_sigclr(batch_size, epochs, checkpoint_file, optimizer):
    if not int(os.environ["SLURM_JOB_NUM_NODES"]) == 1:
        # --gres=gpu:2 == SLURM_GPUS_ON_NODE=2, SLURM_JOB_GPUS=1,2
        # --nodes=2 == SLURM_JOB_NUM_NODES=2, SLURM_NNODES=2
        fabric = Fabric(accelerator="gpu", devices=int(os.environ['SLURM_GPUS_ON_NODE']), num_nodes=int(os.environ['SLURM_JOB_NUM_NODES']))
        print("Training across", int(os.environ['SLURM_JOB_NUM_NODES'])*int(os.environ['SLURM_GPUS_ON_NODE']), 'GPUs')
        fabric.launch(train_internal(batch_size, epochs, checkpoint_file, optimizer))
    else:
        print("Training on a single node")
        train_internal(train_internal(batch_size, epochs, checkpoint_file, optimizer))



if __name__ == "__main__":
    train_sigclr()
