
import h5py
import torch
from torch_geometric.data import Data

def read_ds(path):
    """ Read stellar kinematics from dSph dataset """
    with h5py.File(path, 'r') as f:
        # read dataset attributes
        attrs = dict(f.attrs)

        # read pointer to each graph and concatenate graph features
        ptr = f['ptr'][:]
        features = f['features'][:]
        features = [features[ptr[i]:ptr[i+1]] for i  in range(len(ptr)-1)]

        # read labels
        labels = f['labels'][:]
    return features, labels, attrs

def create_ds(features, labels, transforms):
    dataset = []
    for i in range(len(features)):
        data = Data(
            x=torch.tensor(features[i], dtype=torch.float32),
            y=torch.tensor(labels[i], dtype=torch.float32).reshape(1, -1)
        )
        data = transforms(data)
        dataset.append(data)
    return dataset

