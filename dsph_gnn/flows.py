
from torch import nn, tanh, relu
from pyro.distributions.transforms import affine_autoregressive

from pyknos.nflows import distributions as distributions_
from pyknos.nflows import flows, transforms

def build_iaf(input_dim, num_iafs, iaf_dim):
    iafs = [affine_autoregressive(input_dim, hidden_dims=[iaf_dim]) for _ in range(num_iafs)]
    return iafs

def build_maf(dim=1, num_transforms=8, context_features=None, hidden_features=128):

    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    transforms.MaskedAffineAutoregressiveTransform(
                        features=dim,
                        hidden_features=hidden_features,
                        context_features=context_features,
                        num_blocks=2,
                        use_residual_blocks=False,
                        random_mask=False,
                        activation=tanh,
                        dropout_probability=0.0,
                        use_batch_norm=True,
                    ),
                    transforms.RandomPermutation(features=dim),
                ]
            )
            for _ in range(num_transforms)
        ]
    )

    distribution = distributions_.StandardNormal((dim,))
    neural_net = flows.Flow(transform, distribution)

    return neural_net

