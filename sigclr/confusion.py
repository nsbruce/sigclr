from torchsig.datasets.sig53 import Sig53
import torchsig.transforms as ST
import torch
import os
import click
from sigclr import SigCLR
import numpy as np
import torchsig.transforms as ST
import pickle
from pathlib import Path
import torch.nn as nn
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pytorch_lightning import seed_everything

torch.set_float32_matmul_precision('medium')
num_workers = os.cpu_count()//4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True

@click.group()
def cli():
    pass

@cli.command
@click.option("--weights-file", type=click.Path(path_type=Path))
@click.option("--save-latent-pkl", type=bool, default=False)
@click.option("--save-similarities-npy", type=bool, default=False)
@click.option('--save-html', type=bool, default=True)
@click.option('--show', type=bool, default=False)
def weights_to_confusion(weights_file: Path | None, save_latent_pkl: bool, save_similarities_npy: bool, save_html: bool, show: bool) -> None:

    print('loading dataset')
    root_qa = os.getenv("ROOT_VAL","/project/def-msteve/torchsig/sig53_qa/")
    target_transform = ST.DescToClassIndex(class_list=list(Sig53._idx_to_name_dict.values()))

    # this contains two signals from each class
    qa_dataset = Sig53(
        root=root_qa,
        train=False,
        impaired=False,
        transform=ST.ComplexTo2D(),
        target_transform=target_transform,
        use_signal_data=True,
    )

    if weights_file is not None:
        print('loading model from weights')
        model = SigCLR.load_from_checkpoint(weights_file)
    else:
        print('instantiating fresh model')
        seed_everything(42)
        model = SigCLR(hidden_dim=53, lr=0.001, temperature=0.07, weight_decay=1e-4, batch_size=32, max_epochs=100, device='cuda')
    model.eval()

    # # empty array to fill with the latent vectors output from the network
    # # number of signal classes, number of signals per class, length of latent vector
    # latent_vectors = np.empty((len(qa_dataset)//2, 2, model.encoder.neck_out_features))
    latent_vectors_h1 = dict()
    latent_vectors_h2 = dict()
    latent_vectors_z1 = dict()
    latent_vectors_z2 = dict()

    print(f'building latent vectors')
    count = 0
    for x, y in qa_dataset:
        # Convert x from numpy to torch.Tensor
        x = x.astype(np.float32)
        x_tensor = torch.tensor(x).float().to(model.device)
        
        # Ensure batch dimension is correct (1, C, H, W) or (1, C, L) depending on your data shape
        x_tensor = x_tensor.unsqueeze(0)

        # z is output from the projection head
        # h is output from the encoder
        z, h = model.predict(x_tensor)

        # Detach and move the tensors back to CPU before converting to numpy
        z_np = z.cpu().detach()#.numpy()
        h_np = h.cpu().detach()#.numpy()

        if y not in latent_vectors_h1.keys():
            latent_vectors_h1[y] = h_np
            latent_vectors_z1[y] = z_np
        elif y not in latent_vectors_h2.keys():
            latent_vectors_h2[y] = h_np
            latent_vectors_z2[y] = z_np
        else:
            raise RuntimeError(f'I have too many signals for label {y}')
        
        count += 1
        print(count, len(qa_dataset))
        if count == len(qa_dataset):
            break

    if len(latent_vectors_h1) != len(latent_vectors_h2) != len(latent_vectors_z1) != len(latent_vectors_z2) or len(latent_vectors_h1) != model.encoder.neck_out_features:
        raise RuntimeError(f'I have too few signals {latent_vectors_h1.keys(), latent_vectors_h2.keys(), latent_vectors_z1.keys(), latent_vectors_z2.keys()}')

    if save_latent_pkl:
        with open('latent_vectors.pkl', 'wb') as f:
            pickle.dump((latent_vectors_h1.numpy(), latent_vectors_h2.numpy(), latent_vectors_z1.numpy(), latent_vectors_z2.numpy()), f)

    print('computing similarities')
    similarities_h = np.empty((len(latent_vectors_h1), len(latent_vectors_h2)))
    similarities_z = np.empty((len(latent_vectors_z1), len(latent_vectors_z2)))

    similarity = nn.CosineSimilarity(dim=1)

    for y1, x1 in latent_vectors_h1.items():
        for y2, x2 in latent_vectors_h2.items():
            # similarities[y1,y2] = similarity(torch.Tensor(x1), torch.Tensor(x2))
            similarities_h[y1,y2] = similarity(x1, x2)

    for y1, x1 in latent_vectors_z1.items():
        for y2, x2 in latent_vectors_z2.items():
            # similarities[y1,y2] = similarity(torch.Tensor(x1), torch.Tensor(x2))
            similarities_z[y1,y2] = similarity(x1, x2)

    if save_similarities_npy:
        np.save('similarities-h.npy', similarities_h)
        np.save('similarities-z.npy', similarities_z)

    print('plotting')
    stem = weights_file.stem.replace('-', '_').replace('=','')
    
    categories = list(Sig53._idx_to_name_dict.values())
    fig = make_subplots(rows=1,cols=2, subplot_titles=["h: neck", "z: projection head"], shared_xaxes=True, shared_yaxes=True)
    fig.add_trace(go.Heatmap(z=similarities_h, x=categories, y=categories), row=1, col=1)
    fig.add_trace(go.Heatmap(z=similarities_z, x=categories, y=categories), row=1, col=2)
    fig.update_layout(title_text=stem)
    if save_html:
        fig.write_html(f'confusion-{stem}.html')
    if show:
        fig.show()

@cli.command
@click.option("--npy-file", type=click.Path(path_type=Path))
@click.option('--show', type=bool, default=False)
@click.option('--image-name', type=str, default='confusion.png')
@click.option('--collapse-mod-classes', type=bool, default=False)
@click.option('--collapse-similar-mod-classes', type=bool, default=False)
def similarities_to_collapsed_confusion(npy_file: Path, show: bool, image_name: str, collapse_mod_classes: bool, collapse_similar_mod_classes: bool) -> None:
    SIGCLR_CLASSES: list = [
        "ook",
        "bpsk",
        "4pam",
        "4ask",
        "qpsk",
        "8pam",
        "8ask",
        "8psk",
        "16qam",
        "16pam",
        "16ask",
        "16psk",
        "32qam",
        "32qam_cross",
        "32pam",
        "32ask",
        "32psk",
        "64qam",
        "64pam",
        "64ask",
        "64psk",
        "128qam_cross",
        "256qam",
        "512qam_cross",
        "1024qam",
        "2fsk",
        "2gfsk",
        "2msk",
        "2gmsk",
        "4fsk",
        "4gfsk",
        "4msk",
        "4gmsk",
        "8fsk",
        "8gfsk",
        "8msk",
        "8gmsk",
        "16fsk",
        "16gfsk",
        "16msk",
        "16gmsk",
        "ofdm-64",
        "ofdm-72",
        "ofdm-128",
        "ofdm-180",
        "ofdm-256",
        "ofdm-300",
        "ofdm-512",
        "ofdm-600",
        "ofdm-900",
        "ofdm-1024",
        "ofdm-1200",
        "ofdm-2048",
    ]

    similarities = np.load(npy_file)

    if collapse_mod_classes or collapse_similar_mod_classes:
        group_map = {'fsk':[], 'psk': [], 'ofdm': [], 'msk': [], 'qam': [], 'pam': [], 'ask': [] }
        for mod in SIGCLR_CLASSES:
            for mod_group in group_map.keys():
                if mod_group in mod:
                    # print(mod, mod_group)
                    group_map[mod_group].append(mod)
        
        if collapse_similar_mod_classes:
            group_map['fsk'].extend(group_map['msk'])
            del group_map['msk']
            group_map['ask'].extend(group_map['pam'])
            del group_map['pam']
            group_map['ask'].extend(group_map['qam'])
            del group_map['qam']
            # group_map['ask'].extend(group_map['ofdm'])
            # del group_map['ofdm']
        
        # print(group_map)

        # Mapping of original labels to group index
        grouped_indexes = {key: [] for key in group_map}
        for i, label in enumerate(SIGCLR_CLASSES):
            for group, members in group_map.items():
                if label in members:
                    grouped_indexes[group].append(i)
        
        # print(grouped_indexes)
        
        # Initialize the simplified matrix
        num_groups = len(group_map)
        simplified_matrix = np.zeros((num_groups, num_groups), dtype=float)
        group_labels = list(group_map.keys())


        # Populate the simplified matrix
        for i, group_row in enumerate(group_labels):
            for j, group_col in enumerate(group_labels):
                # Sum over the original indexes for each group pair
                divide_by_count = 0
                for row_idx in grouped_indexes[group_row]:
                    for col_idx in grouped_indexes[group_col]:
                        # print('output_cell: ', group_row, group_col, 'input_val: ', similarities[row_idx, col_idx])
                        simplified_matrix[i, j] += similarities[row_idx, col_idx]
                        divide_by_count += 1
                simplified_matrix[i, j] /= divide_by_count
        tick_labels = list(group_map.keys())
        colorbar_title='Mean similarity'
    else:
        simplified_matrix = similarities
        tick_labels = SIGCLR_CLASSES
        colorbar_title='Similarity'

    fig = go.Figure(go.Heatmap(
        z=simplified_matrix,
        x=tick_labels,
        y=tick_labels,
        colorscale='Viridis',
        colorbar=dict(
            title=colorbar_title,
            titleside="right",
            ticks="outside"
        )
    ))
    fig.update_layout(
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            title='Modulation A',
            ticks="outside",
            showgrid=False,
        ),
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            title='Modulation B',
            ticks="outside",
            showgrid=False
            ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Times New Roman",
            size=14,
            color='black'
        ),
        width=615,
        height=600,
        # margin=dict(l=50, r=50, t=0, b=0)
    )
    if show:
        fig.show()
    else:
        fig.write_image(image_name, scale=3)



if __name__ == "__main__":
    cli()
