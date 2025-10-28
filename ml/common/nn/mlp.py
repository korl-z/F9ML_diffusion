import logging

import torch
import torch.nn as nn


class MLPLayer(nn.Module):
    def __init__(self, n_in, n_out, act, batchnorm, act_first, dropout):
        super().__init__()
        self.layer = nn.ModuleList()

        if act is not None:
            self.act_func = getattr(nn, act)
        else:
            self.act_func = None

        if act_first:
            if dropout:
                self.layer.append(nn.Dropout(dropout))

            if self.act_func is not None:
                self.layer.append(self.act_func())

            if batchnorm:
                self.layer.append(nn.BatchNorm1d(n_in))

            self.layer.append(nn.Linear(n_in, n_out))

        else:
            if dropout:
                self.layer.append(nn.Dropout(dropout))

            if batchnorm:
                self.layer.append(nn.BatchNorm1d(n_in))

            if self.act_func is not None:
                self.layer.append(self.act_func())

            self.layer.append(nn.Linear(n_in, n_out))

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x


class BasicMLP(nn.Module):
    def __init__(self, layers, act="ReLU", act_out=None, batchnorm=False, act_first=False, dropout=None):
        super().__init__()
        # note that act_first does not apply if batchnorm is False
        self.layer_dims = layers
        self.layers = nn.ModuleList()

        self.layers.add_module("input_layer", MLPLayer(layers[0], layers[1], None, False, False, dropout))

        for i in range(1, len(layers) - 2):
            n_in, n_out = layers[i], layers[i + 1]
            layer = MLPLayer(n_in, n_out, act, batchnorm, act_first, dropout)
            self.layers.add_module(f"layer{i}", layer)

        self.layers.add_module("final_layer", MLPLayer(layers[-2], layers[-1], act, batchnorm, act_first, dropout))

        if act_out is not None:
            self.layers.add_module("act_out", getattr(nn, act_out)())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResMLPBlock(nn.Module):
    def __init__(self, n, act, batchnorm=True, act_first=True, dropout=None, repeats=2):
        super().__init__()

        self.blocks = nn.ModuleList()

        for i in range(repeats):
            block = MLPLayer(n, n, act, batchnorm, act_first, dropout)
            self.blocks.add_module(f"block{i}", block)

    def forward(self, x):
        z = x
        for block in self.blocks:
            x = block(x)
        return x + z


class BasicResMLP(nn.Module):
    def __init__(self, layers, act="ReLU", act_out=None, batchnorm=True, act_first=False, dropout=None, repeats=2):
        """Identity Mappings in Deep Residual Networks: https://arxiv.org/abs/1603.05027"""
        super().__init__()
        self.layers = nn.ModuleList()

        self.layers.add_module("input_layer", nn.Linear(layers[0], layers[1]))

        for i in range(1, len(layers) - 2):
            n_in, n_out = layers[i], layers[i + 1]
            if n_in == n_out:
                layer = ResMLPBlock(
                    n_in, act, batchnorm=batchnorm, act_first=act_first, dropout=dropout, repeats=repeats
                )
                self.layers.add_module(f"layer{i}", layer)
            else:
                logging.warning("Output and input dimensions do not match in ResNet, adding a linear layer!")
                layer = MLPLayer(n_in, n_out, act, batchnorm=batchnorm, act_first=act_first, dropout=dropout)
                self.layers.add_module(f"layer{i}", layer)

        self.layers.add_module("final_layer", MLPLayer(layers[-2], layers[-1], act, batchnorm, act_first, dropout))

        if act_out is not None:
            self.layers.add_module("act_out", getattr(nn, act_out)())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DiscriminatorMLP(nn.Module):
    """
    A simple MLP discriminator to distinguish between real uniform data
    and the generator's output.

    Takes flattened data of shape (batch_size, n_features) and outputs
    a single logit (batch_size, 1).
    """
    def __init__(self, n_features, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        if x.ndim > 2:
            x = x.flatten(start_dim=1)
        return self.net(x)
    
if __name__ == "__main__":
    from ml.common.nn.netron_vis import visualize_model

    mlp = BasicMLP([20, 128, 128, 128, 1], act_out="Sigmoid", batchnorm=True, act_first=False, dropout=0.5)
    x = torch.randn(1024, 20)

    # TODO: code this in a more straightforward way
    res_mlp = BasicResMLP(
        [32, 128, 128, 128, 24], act_out="Sigmoid", batchnorm=True, act_first=False, dropout=0.5, repeats=2
    )
    x = torch.randn(1024, 32)

    print(res_mlp)
    visualize_model(res_mlp, x)
