from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning import Trainer, seed_everything
# from lightning.pytorch.strategies import DDPStrategy
from pytorch_lightning.strategies import DDPStrategy
from torchsig.datasets.sig53 import Sig53
from torch.utils.data import DataLoader
import torchsig.transforms as ST
import torch
import os
import datetime
import click
from dataset import SigCLRDataset
from sigclr import SigCLR


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
CHECKPOINT_PATH = f"./saved_models_{runID}/"
root_train = os.getenv("ROOT_TRAIN")#,"/project/def-msteve/torchsig/sig53/")
root_val = os.getenv("ROOT_VAL")#,"/project/def-msteve/torchsig/sig53/") 

def setup():
    torch.set_float32_matmul_precision('medium')
    num_workers = os.cpu_count()//4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True

    print(f"Using device: {device}")
    print(f"Number of workers: {num_workers}")


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
    # Instantiate the Sig53 Validation Dataset
    train = False
    sig53_val = SigCLRDataset(Sig53(
        root=root_val, 
        train=train, 
        impaired=impaired,
        transform=None,
        target_transform=target_transform,
        use_signal_data=True,
    ),transforms=contrast_transforms)

    print(f'Our validation data comes from {root_val}, and has {len(sig53_val)} impaired signals')

    return sig53_train, sig53_val


@click.command()
@click.option('--batch_size', default=32, help='Batch size used during training and validation.')
@click.option('--restart', default=1, help='Restart from a previousely trained model.')
@click.option('--device', default='cuda', help='What device to use for training.')
@click.option('--num_workers', default=4, help='The number of works.')
@click.option('--hidden_dim', default=53, help='Dimension of the hidden layer.')
@click.option('--epochs', default=100, help='Number of epochs during training.')
@click.option('--lr', default=0.001, help='Learning rate for the optimizer.')
@click.option('--weight_decay', default=1e-4, help='Weight_decay for the optimizer.')
@click.option('--temperature', default=0.07, help='Temperature rate used for ntXent loss computation.')
@click.option('--val_every', default=10, help='Run validation every val_every epochs.')
@click.option('--ckpt_file', default='last.ckpt', help='Restart from a previous checkpointed model. Provide the file ')
def train_sigclr(hidden_dim, lr, temperature, weight_decay, batch_size, epochs, device, val_every, restart, ckpt_file ,num_workers):

    sig53_train, sig53_val = setup()

    
    accel="gpu" if str(device) == "cuda" else "cpu"
    checkpoint_callback = ModelCheckpoint(dirpath=CHECKPOINT_PATH, every_n_epochs=1, mode="min", 
                                monitor="val_loss", save_top_k=3,save_last=True)
    ddp = DDPStrategy(process_group_backend="nccl",timeout=datetime.timedelta(seconds=5400))
    trainer = Trainer(
        default_root_dir=CHECKPOINT_PATH,
        devices=-1,
        accelerator=accel,
        max_epochs=epochs,
        strategy=ddp,#'ddp',
        #check_val_every_n_epoch=val_every,
        val_check_interval=100,
        enable_progress_bar=False,
        num_nodes=int(os.environ.get("SLURM_JOB_NUM_NODES","1")),
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor("epoch"),
            # EarlyStopping(monitor="val_loss", mode="min", patience=100, verbose=False)
        ],
        # progress_bar_refresh_rate=1,
        sync_batchnorm=True
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    train_loader = DataLoader(
            sig53_train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=num_workers,
        )
    val_loader = DataLoader(
            sig53_val,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=num_workers,
        )
    # If restart and the pretrained model exists, load it and train some more.
    if restart and os.path.isfile(ckpt_file):
        print(f"Found pretrained model at {ckpt_file}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = SigCLR.load_from_checkpoint(ckpt_file)
        trainer.fit(model, train_loader, val_loader,ckpt_path=ckpt_file)
        # Load best checkpoint after training
        model = SigCLR.load_from_checkpoint(checkpoint_callback.best_model_path)
    else:
        seed_everything(42)  # To be reproducable
        model = SigCLR(hidden_dim=hidden_dim, lr=lr, temperature=temperature, weight_decay=weight_decay, batch_size=batch_size, max_epochs=epochs, device=device)
        trainer.fit(model, train_loader, val_loader)
        # Load best checkpoint after training
        model = SigCLR.load_from_checkpoint(checkpoint_callback.best_model_path)

    return model

if __name__ == "__main__":
    sigclr_model = train_sigclr()
    # what do more with the sigclr_model here as it is the best model selected.

