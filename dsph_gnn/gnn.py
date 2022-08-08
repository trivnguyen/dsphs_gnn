
import torch
import torch_geometric
import torchmetrics
from torch_geometric.nn import ChebConv, GATConv, GCNConv

from .flows import build_maf

class DeepSet(torch.nn.Module):

    def __init__(self, *args, **kargs):
        super().__init__()
        self.layer = torch.nn.Linear(*args, **kargs)

    def forward(self, x, edge_index, edge_weight=None):
        return self.layer(x)

class GNNRegressor(torch.nn.Module):
    """ GNN Regressor model"""

    conv_dict = {
        'CHEB': ChebConv,
        'GAT': GATConv,
        'GCN': GCNConv,
        'DEEPSET': DeepSet,
    }
    conv_edge_weight = {
        'CHEB': True,
        'GAT': False,
        'GCN': True,
        'DEEPSET': False,
    }
    conv_default_kargs = {
        'CHEB': dict(K=4, normalization='sym', bias=True),
        'GAT': dict(heads=2),
        'GCN': dict(),
        'DEEPSET': dict(),
    }

    def __init__(self,
        in_channels, out_channels, conv_name, conv_hparams,
        num_layers, hidden_channels, num_layers_fc, hidden_channels_fc,
        hidden_channels_flows, num_transforms):
        """
        Arguments:
        - in_channels: [int] number of input channels
        - out_channels: [int] number of output channels
        - conv_name [str]: name of the GNN layer
        - conv_hparams: [dict] dictionary with extra kargs for GNN layer
        - num_layers: [int] number of GNN hidden layers
        - hidden_channels: [int] number of GNN hidden channels
        - num_layers_fc: [int] number of FC hidden layers
        - hidden_channels_fc: [int] number of FC hidden channels
        - hidden_channels_flows: [int] number of NF hidden channels
        - num_transforms: [int] number of NF transformations
        """

        super().__init__()

        self.conv_name = conv_name

        if conv_name in self.conv_dict:
            self.conv_layer = self.conv_dict[conv_name]
        else:
            raise KeyError(
                f"Unknown model name \"{conv_name}\"."\
                f"Available models are: {str(self.conv_dict.keys())}")

        # GNN layers
        self.conv_layers = torch.nn.ModuleList()
        default_conv_hparams = self.conv_default_kargs[conv_name]
        default_conv_hparams.update(conv_hparams)
        for i in range(num_layers):
            n_in = in_channels if i == 0 else hidden_channels
            n_out = hidden_channels
            self.conv_layers.append(
                self.conv_layer(n_in, n_out, **default_conv_hparams))

        # FC layers
        self.fc_layers = torch.nn.ModuleList()
        for i in range(num_layers_fc):
            n_in = hidden_channels if i == 0 else hidden_channels_fc
            n_out = hidden_channels_fc
            self.fc_layers.append(torch.nn.Linear(n_in, n_out))

        # Create MAF layers
        self.maf = build_maf(
            dim=out_channels, num_transforms=num_transforms,
            context_features=hidden_channels_flows,
            hidden_features=hidden_channels_flows)

    def forward(self, x, edge_index, batch, edge_weight=None):
        """ Forward pass function """

        # Apply GNN layers
        for i in range(len(self.conv_layers)):
            if self.conv_edge_weight[self.conv_name]:
                x = self.conv_layers[i](x, edge_index, edge_weight=edge_weight)
            else:
                x = self.conv_layers[i](x, edge_index)
            x = torch.nn.functional.relu(x)

        # Global mean pooling to average features across node, then pass into readout layer
        x = torch_geometric.nn.global_mean_pool(x, batch)

        # Apply FC layers on features
        for i in range(len(self.fc_layers)):
            x = self.fc_layers[i](x)
            if i != len(self.fc_layers) - 1:
                x = torch.nn.functional.relu(x)
        return x

