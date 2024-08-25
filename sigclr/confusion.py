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
import csv

torch.set_float32_matmul_precision('medium')
num_workers = os.cpu_count()//4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True

# @click.group()
# def cli():
#     pass

# @cli.command()
# @click.option("--weights-file", type=click.Path())
# @click.option("--save-latent-pkl", type=bool, default=True)
# def weights_to_latent(weights_file:Path, save_latent_pkl: bool) -> tuple[np.ndarray, np.ndarray]:

#     ## get data to compute confusion on
#     root_qa = os.getenv("ROOT_VAL","/project/def-msteve/torchsig/sig53_qa/")
#     target_transform = ST.DescToClassIndex(class_list=list(Sig53._idx_to_name_dict.values()))

#     # this contains two signals from each class
#     qa_dataset = Sig53(
#         root=root_qa,
#         train=False,
#         impaired=False,
#         transform=ST.ComplexTo2D(),
#         target_transform=target_transform,
#         use_signal_data=True,
#     )


#     model = SigCLR.load_from_checkpoint(weights_file)
#     model.eval()

#     # # empty array to fill with the latent vectors output from the network
#     # # number of signal classes, number of signals per class, length of latent vector
#     # latent_vectors = np.empty((len(qa_dataset)//2, 2, model.encoder.neck_out_features))
#     latent_vectors_1 = dict()
#     latent_vectors_2 = dict()

#     # print(f'Created an empty array of shape {latent_vectors.shape}')
#     count = 0
#     for x, y in qa_dataset:
#         # Convert x from numpy to torch.Tensor
#         x = x.astype(np.float32)
#         x_tensor = torch.tensor(x).float().to(model.device)
        
#         # Ensure batch dimension is correct (1, C, H, W) or (1, C, L) depending on your data shape
#         x_tensor = x_tensor.unsqueeze(0)

#         # z is output from the projection head
#         # h is output from the encoder
#         z, h = model.predict(x_tensor)

#         # Detach and move the tensors back to CPU before converting to numpy
#         # z_np = z.cpu().detach().numpy()
#         h_np = h.cpu().detach().numpy()

#         if y not in latent_vectors_1.keys():
#             latent_vectors_1[y] = h_np
#         elif y not in latent_vectors_2.keys():
#             latent_vectors_2[y] = h_np
#         else:
#             raise RuntimeError(f'I have too many signals for label {y}')
        
#         count += 1
#         print(count, len(qa_dataset))
#         if count == len(qa_dataset):
#             break


#     if len(latent_vectors_1) != len(latent_vectors_2) or len(latent_vectors_1) != model.encoder.neck_out_features:
#         raise RuntimeError(f'I have too few signals {latent_vectors_1.keys(), latent_vectors_2.keys()}')

#     if save_latent_pkl:
#         with open('latent_vectors.pkl', 'wb') as f:
#             pickle.dump((latent_vectors_1, latent_vectors_2), f)
    
#     return (latent_vectors_1, latent_vectors_2)


# @cli.command
# @click.option("--latent-vectors-pkl", type=click.Path())
# @click.option("--save-similarities-npy", type=bool, default=True)
# def latent_to_similarities(latent_vectors_pkl: Path, save_similarities_npy: bool) -> np.ndarray:
#     with open(latent_vectors_pkl, 'rb') as f:
#         latent_vectors_1, latent_vectors_2 = pickle.load(f)
#     similarities = np.empty((len(latent_vectors_1), len(latent_vectors_2)))

#     similarity = nn.CosineSimilarity(dim=1)

#     for y1, x1 in latent_vectors_1.items():
#         for y2, x2 in latent_vectors_2.items():
#             similarities[y1,y2] = similarity(torch.Tensor(x1), torch.Tensor(x2))

#     if save_similarities_npy:
#         np.save('similarities.npy', similarities)

#     return similarities

# @cli.command
# @click.option('--similarities-npy', type=click.Path())
# @click.option('--save-html', type=bool, default=True)
# @click.option('--show', type=bool, default=False)
# def similarities_to_confusion(similarities_npy:Path, save_html: bool, show: bool) -> None:
#     similarities=np.load(similarities_npy)
#     fig = px.imshow(similarities)
#     if save_html:
#         fig.write_html('confusion.html')
#     if show:
#         fig.show()

@click.command
@click.option("--weights-file", type=click.Path(path_type=Path))
@click.option("--save-latent-pkl", type=bool, default=False)
@click.option("--save-similarities-npy", type=bool, default=False)
@click.option('--save-html', type=bool, default=True)
@click.option('--show', type=bool, default=False)
def weights_to_confusion(weights_file: Path, save_latent_pkl: bool, save_similarities_npy: bool, save_html: bool, show: bool) -> None:

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

    print('loading model')
    model = SigCLR.load_from_checkpoint(weights_file)
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
        np.save('similarities.npy', np.concat(similarities_h, similarities_z))

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

if __name__ == "__main__":
    weights_to_confusion()