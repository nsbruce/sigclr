# from train import contrast_transforms
import os
from torchsig.datasets.sig53 import Sig53
import torchsig.transforms as ST
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy import signal as sp
import numpy as np
import click
import re


# key: index is data index: augmentation
apply_what_where = {
    2: ST.RandomPhaseShift(), # 2gmsk
    2: ST.Clip(), # 2gmsk
    2: ST.TimeVaryingNoise(), # 2gmsk
    2: ST.TimeReversal(), # 2gmsk
    64: ST.RandomTimeShift(100), # 2msk
}

@click.command
@click.option('--data-index', type=int, default=0)
def plot(data_index: int):
    root_qa = os.getenv("ROOT_VAL","/project/def-msteve/torchsig/sig53_qa/")

    qa_dataset = Sig53(
        root=root_qa,
        train=False,
        impaired=False,
    )

    contrast_transforms = [
        ST.TimeVaryingNoise(-80, -15, 4, True, False),
        ST.RandomPhaseShift(-0.5),
        ST.TimeReversal(True),
        ST.Clip(0.75),
        ST.RandomTimeShift(500),
        ST.GainDrift(0.5, 0.49, 0.1),
        ST.LocalOscillatorDrift(0.05,0.01),
        ST.SpectralInversion(),
    ]


    x,y = qa_dataset.__getitem__(data_index)

    subplot_titles=[t.__class__.__name__ for t in contrast_transforms]

    # split titles up
    subplot_titles = [re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', s) for s in subplot_titles]
    subplot_titles.insert(0,"Original")
    print(subplot_titles)

    fig = make_subplots(cols=3, rows=3, subplot_titles=subplot_titles, horizontal_spacing=0.005, vertical_spacing=0.05, shared_xaxes=True, shared_yaxes=True)

    transparent_black = "rgba(0,0,0,0.25)"
    axis_layout = dict(
        showgrid=True,
        zeroline=True,
        gridwidth=1,
        gridcolor=transparent_black,
        zerolinewidth=1,
        zerolinecolor=transparent_black,
        mirror=True,
        showline=True,
        ticks="inside",
        linecolor="black",
        linewidth=2,
        showticklabels=False,
        title_font_size=15,
    )

    contrast_transforms.insert(0,ST.Identity())  # dummy one to make space for the original
    for i, transform in enumerate(contrast_transforms):
        print(f'on {i+1}/{len(contrast_transforms)}')
        x0 = x.copy()
        x1 = transform(x0)

        row = i // 3 + 1
        col = i % 3 +1

        print('row',row, 'col',col)

        fig.add_trace(go.Scatter(y=np.real(x1[1024:-1024]), line_color='red', name='Re.', showlegend=True if row == 1 and col == 1 else False), row=row, col=col)
        fig.add_trace(go.Scatter(y=np.imag(x1[1024:-1024]), line_color='blue', name='Im.', showlegend=True if row == 1 and col == 1 else False), row=row, col=col)

        fig.update_xaxes(title_text="Time" if row == 3 else None, **axis_layout, row=row, col=col)
        fig.update_yaxes(title_text="Amplitude" if col == 1 else None,**axis_layout, row=row, col=col)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        coloraxis=dict(
            colorscale='viridis'
        ),
        legend=dict(
            orientation='h',
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
            font=dict(
                size=16
            )
        ),
        font=dict(
            family="Times New Roman",
            size=12,
            color='black'
        ),
        # xaxis_title=dict(font=dict(size=12)),
        # yaxis_title=dict(font=dict(size=12)),
        # title=dict(font=dict(size=12)),
        
    )

    fig.write_html(f'transforms-{data_index}.html')
    fig.write_image("transforms.svg", width=900, height=700, scale=1.5)
if __name__ == '__main__':
    plot()