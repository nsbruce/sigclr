from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader
import torchsig.transforms as ST
import torch
import os
import click
from sigclr.dataset import SigCLRNarrowbandDataset
from sigclr.sigclr2 import SigCLR
from sigclr.modulation_classes import SIGCLR_CLASSES

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

assert torch.cuda.is_available()
assert int(os.environ.get("SLURM_JOB_NUM_NODES","1")) == 1

CHECKPOINT_PATH=os.getenv("CHECKPOINT_PATH")
root_train = os.getenv("ROOT_TRAIN")
root_val = os.getenv("ROOT_VAL")


def setup_datasets(impaired: bool, batch_size: int):
    torch.set_float32_matmul_precision('medium')
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
    print(f'Our training data comes from {root_train}, and has {len(train_dataset)} signals')
    # Instantiate the validation dataset
    val_dataset = SigCLRNarrowbandDataset(
        root=root_val, 
        train=False, 
        impaired=impaired,
        target_transform=target_transform,
        use_signal_data=True,
        transforms=contrast_transforms
    )

    print(f'Our validation data comes from {root_val}, and has {len(val_dataset)} signals')

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


@click.command()
@click.option('--batch-size', default=32, help='Batch size used during training and validation.')
@click.option('--epochs', default=100, help='Number of epochs during training.')
@click.option('--num-workers', default=4, help='The number of workers.')
@click.option('--checkpoint-file', help='Restarts from the provided previous checkpointed model file.')
def train_sigclr(batch_size, epochs, num_workers, checkpoint_file):

    lr=0.001  # for optimizer
    hidden_dim=256  # dimension of the hidden layer
    weight_decay=1e-4  # for optimizer
    temperature=0.07  # for ntXent loss computation
    
    train_loader, val_loader = setup_datasets(impaired=False, batch_size=batch_size)

    
    checkpoint_callback = ModelCheckpoint(dirpath=CHECKPOINT_PATH, every_n_epochs=1, mode="min", monitor="val_loss", save_top_k=3,save_last=True)

    trainer = Trainer(
        default_root_dir=CHECKPOINT_PATH,
        devices="auto",
        accelerator="gpu",
        max_epochs=epochs,
        enable_progress_bar=False,
        callbacks=checkpoint_callback,
    )

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
        model = SigCLR(hidden_dim=hidden_dim, lr=lr, temperature=temperature, weight_decay=weight_decay, batch_size=batch_size, max_epochs=epochs, device=torch.device("cuda"), num_encoder_output_features=64)
        trainer.fit(model, train_loader, val_loader)

    return model

if __name__ == "__main__":
    sigclr_model = train_sigclr()
    # what do more with the sigclr_model here as it is the best model selected.
