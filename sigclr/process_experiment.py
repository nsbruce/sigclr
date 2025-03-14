from pathlib import Path
import numpy as np
import pandas as pd
from sigclr.metrics import evaluate_similarity_matrix


experiment = {
    'adamW': [64,128,256,512,1024,2048],
    'lars': [1024,2048,4096]
}

saved_models = Path(__file__).parent.parent / 'saved_models'

df = pd.DataFrame(columns=['adamW_h', 'adamW_z', 'lars_h', 'lars_z'])
df['batch_sizes'] = sorted(set(list(experiment.values())[0] + list(experiment.values())[1]))
df = df.set_index('batch_sizes')
print(df)

for optim, batch_sizes in experiment.items():
    for batch_size in batch_sizes:
        print('Loading', saved_models / f'{optim}-single-node' / f'batchsize-{batch_size}-samples-512' / 'similarities-h-epoch100.npy')
        similarities_h = np.load(saved_models / f'{optim}-single-node' / f'batchsize-{batch_size}-samples-512' / 'similarities-h-epoch100.npy')
        similarities_z = np.load(saved_models / f'{optim}-single-node' / f'batchsize-{batch_size}-samples-512' / 'similarities-z-epoch100.npy')

        h_val = evaluate_similarity_matrix(similarities_h)
        z_val = evaluate_similarity_matrix(similarities_z)

        print('h_val', h_val, 'z_val', z_val)
        df.loc[batch_size, f'{optim}_h'] = h_val
        df.loc[batch_size, f'{optim}_z'] = z_val

print(df)
df.to_csv('similarity-evals-100-epochs.csv')
