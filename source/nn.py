import torch
import torch.nn as nn
from torchvision.ops import MLP

from .layers import GraphAggregation, GraphConvolution, MultiDenseLayers
from .quantum import NoiseQuantumGenerator, PatchQuantumGenerator


class Generator(nn.Module):
    """Generator network of MolGAN"""

    def __init__(
        self,
        dataset,
        *,
        conv_dims=[128, 256, 512],
        z_dim=8,
        dropout=0.0,
    ):
        super(Generator, self).__init__()
        self.dataset = dataset
        self.conv_dims = conv_dims
        self.z_dim = z_dim
        self.dropout = dropout

        self.vertexes = self.dataset.num_vertices
        self.edges = self.dataset.bond_num_types
        self.nodes = self.dataset.atom_num_types

        self.multi_dense_layers = MLP(
            self.z_dim,
            self.conv_dims,
            activation_layer=nn.Tanh,
            dropout=self.dropout,
        )
        self.edges_layer = nn.Linear(
            self.conv_dims[-1], self.edges * self.vertexes * self.vertexes
        )
        self.nodes_layer = nn.Linear(self.conv_dims[-1], self.vertexes * self.nodes)
        self.dropout_layer = nn.Dropout(self.dropout)

    def _generate_z(self, batch_size):
        return torch.rand(batch_size, self.z_dim)

    def forward(self, batch_size):
        z = self._generate_z(batch_size)
        output = self.multi_dense_layers(z)
        edges_logits = self.edges_layer(output).view(
            -1, self.edges, self.vertexes, self.vertexes
        )
        edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2
        edges_logits = self.dropout_layer(edges_logits.permute(0, 2, 3, 1))

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropout_layer(
            nodes_logits.view(-1, self.vertexes, self.nodes)
        )

        return edges_logits, nodes_logits


class QuantumGenerator(nn.Module):
    def __init__(
        self,
        dataset,
        *,
        noise_num_qubits=8,
        noise_depth=3,
        patch_num_generators=45,
    ):
        super(QuantumGenerator, self).__init__()
        self.dataset = dataset

        self.noise_num_qubits = noise_num_qubits
        self.noise_depth = noise_depth
        self.patch_num_generators = patch_num_generators

        self.noise_generator = NoiseQuantumGenerator(
            self.noise_num_qubits, self.noise_depth
        )
        self.patch_generator = PatchQuantumGenerator(self.patch_num_generators)

    def forward(self, batch_size):
        noise = self.noise_generator(batch_size)
        return self.patch_generator(noise)


class Discriminator(nn.Module):
    """Discriminator network of MolGAN"""

    def __init__(
        self,
        dataset,
        *,
        conv_dims=[[128, 64], 128, [128, 64]],
        with_features=False,
        f_dim=0,
        dropout=0.0,
    ):
        super(Discriminator, self).__init__()
        self.dataset = dataset
        self.conv_dims = conv_dims
        self.with_features = with_features
        self.f_dim = f_dim
        self.dropout = dropout

        self._initialize()

    def _initialize(self):
        m_dim = self.dataset.atom_num_types
        b_dim = self.dataset.bond_num_types - 1
        self.activation_f = nn.Tanh()
        graph_conv_dim, aux_dim, linear_dim = self.conv_dims
        self.gcn_layer = GraphConvolution(
            m_dim, graph_conv_dim, b_dim, self.with_features, self.f_dim, self.dropout
        )
        self.agg_layer = GraphAggregation(
            graph_conv_dim[-1] + m_dim,
            aux_dim,
            self.activation_f,
            self.with_features,
            self.f_dim,
            self.dropout,
        )
        self.multi_dense_layers = MultiDenseLayers(
            aux_dim,
            linear_dim,
            self.activation_f,
            self.dropout,
        )
        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adjacency_tensor, hidden, node, activation=None):
        adj = adjacency_tensor[:, :, :, 1:].permute(0, 3, 1, 2)
        h = self.gcn_layer(node, adj, hidden)
        h = self.agg_layer(node, h, hidden)

        h = self.multi_dense_layers(h)

        output = self.output_layer(h)
        output = activation(output) if activation is not None else output

        return output, h
