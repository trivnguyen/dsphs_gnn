
import torch
import pytorch_lightning as pl
import torch_geometric.transforms as T

class FeaturePreprocess(T.BaseTransform):
    def __init__(self, npos=2, log_radius=True):
        self.npos = npos
        self.log_radius = log_radius

    def __call__(self, data):
        pos, vel = data.x[:, :self.npos], data.x[:, self.npos:]
        radius = torch.linalg.norm(pos, ord=2, dim=1, keepdims=True)

        if self.log_radius:
            data.x = torch.stack([torch.log10(radius), vel], axis=1).squeeze()
        else:
            data.x = torch.stack([radius, vel], axis=1).squeeze()
        data.pos = pos
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}()')


class EdgeWeightDistance(T.BaseTransform):
    def __init__(self, norm=False):
        self.norm = False

    def __call__(self, data):
        x1 = data.pos[data.edge_index[0]]
        x2 = data.pos[data.edge_index[1]]
        d = torch.linalg.norm(x1-x2, ord=2, dim=1)
        if self.norm:
            rho = torch.mean(d)
            data.edge_weight = torch.exp(-d**2 / rho**2)
        else:
            data.edge_weight = torch.exp(-d**2)
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm})')

class GraphTransforms(T.BaseTransform):
    graph_dict = {
        "KNN": T.KNNGraph,
        "RADIUS": T.RadiusGraph,
    }
    def __init__(self,
        graph_name, graph_hparams, feature_hparams={},
        edge_weight=False, edge_weight_norm=False):

        self.transforms = []
        self.transforms.append(FeaturePreprocess(**feature_hparams))
        if graph_name in self.graph_dict:
            self.transforms.append(self.graph_dict[graph_name](**graph_hparams))
        else:
            raise KeyError(
                f"Unknown graph name \"{graph_name}\"."\
                f"Available models are: {str(self.graph_dict.keys())}")
        if edge_weight:
            self.transforms.append(EdgeWeightDistance(edge_weight_norm))
        self.transforms = T.Compose(self.transforms)

    def __call__(self, data):
        return self.transforms(data)

