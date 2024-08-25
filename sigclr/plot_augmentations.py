# from train import contrast_transforms
import os
from torchsig.datasets.sig53 import Sig53
import torchsig.transforms as ST
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy import signal as sp
import numpy as np
import click

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
        ST.TimeVaryingNoise(-80, -20, 4, True, False),
        ST.RandomPhaseShift(-1),
        ST.TimeReversal(True),
        ST.Clip(0.75),
        # ST.TimeCrop(),
        ST.RandomTimeShift(100),
        ST.GainDrift(0.1, 0.015, 0.01),
        ST.LocalOscillatorDrift(0.1,0.01),
        ST.SpectralInversion(),
    ]


    x,y = qa_dataset.__getitem__(data_index)

    def get_spectrogram(x):
        _, _, spectrogram = sp.spectrogram(
                x=x,
                fs=1,#self.sample_rate,
                window=sp.windows.tukey(256,0.25),#self.window,
                nperseg=256,#self.nperseg,
                noverlap=None, #self.noverlap,
                nfft=None,#self.nfft,
                return_onesided=False,
        )
        spectrogram = 20 * np.log10(np.fft.fftshift(np.abs(spectrogram), axes=0))
        return spectrogram

    spectrogram1 = get_spectrogram(x)
    zmin=spectrogram1.min()
    zmax=spectrogram1.max()
    subplot_titles=[t.__class__.__name__ for t in contrast_transforms]
    subplot_titles.extend([""]*3*len(contrast_transforms))
    print(subplot_titles)

    fig = make_subplots(cols=len(contrast_transforms), rows=4, subplot_titles=subplot_titles, horizontal_spacing=0.005, vertical_spacing=0.005)#, shared_xaxes=True) # time, augtime, freq augfreq

    for i, transform in enumerate(contrast_transforms):
        print(f'on {i+1}/{len(contrast_transforms)}')
        x0 = x.copy()
        fig.add_trace(go.Scatter(y=np.real(x0), line_color='red'), row=1, col=i+1)
        fig.add_trace(go.Scatter(y=np.imag(x0), line_color='blue'), row=1, col=i+1)
        fig.add_trace(go.Heatmap(z=spectrogram1, coloraxis='coloraxis'), row=3, col=i+1)

        x1 = transform(x0)

        fig.add_trace(go.Scatter(y=np.real(x1), line_color='red'), row=2, col=i+1)
        fig.add_trace(go.Scatter(y=np.imag(x1), line_color='blue'), row=2, col=i+1)
        spectrogram2 = get_spectrogram(x1)
        fig.add_trace(go.Heatmap(z=spectrogram2, coloraxis='coloraxis'), row=4, col=i+1)

        # Ensure consistent color scale for heatmaps
        zmin = min(zmin, spectrogram2.min())
        zmax = max(zmax, spectrogram2.max())

    fig.update_traces(zmin=zmin, zmax=zmax, selector=dict(type='heatmap'))

    # second row
    for i in range(9, 17):
        fig.update_xaxes(matches=f'x{i-8}', row=2, col=i%8)
        fig.update_yaxes(matches=f'y{i-8}', row=2, col=i%8)
    fig.update_xaxes(matches='x', row=2, col=1)
    fig.update_yaxes(matches='y', row=2, col=1)

    # fourth row
    for i in range(25,33):
        fig.update_xaxes(matches=f'x{i-8}', row=4, col=i%24)
        fig.update_yaxes(matches=f'y{i-8}', row=4, col=i%24)

    for i in range(1, len(contrast_transforms)+1):
        for j in range(1,5):
            if i != 0:
                fig.update_yaxes(showticklabels=False, row=j, col=i)
            else:
                fig.update_yaxes(showticklabels=True, row=j, col=i)
        for j in range(1,4):
            fig.update_xaxes(showticklabels=False, row=j, col=i)

    fig.update_layout(title_text=f'data class {y[0]} ({Sig53.convert_idx_to_name(y[0])}), index {data_index}', showlegend=False, coloraxis=dict(colorscale='viridis'))

    fig.write_html(f'transforms-{data_index}.html')

if __name__ == '__main__':
    plot()