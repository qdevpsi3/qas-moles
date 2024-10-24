import random

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from torch import nn
from torchvision.ops import MLP

from .layers import GraphAggregation, GraphConvolution, MultiDenseLayers


class QuantumMolGanNoise(nn.Module):
    def __init__(self, num_qubits: int = 8, num_layers: int = 3):
        super(QuantumMolGanNoise, self).__init__()

        # Register the parameters with the module
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        # Initialize weights with PyTorch (between -pi and pi) and register as a learnable parameter
        self.weights = nn.Parameter(
            torch.rand(num_layers, (num_qubits * 2 - 1)) * 2 * torch.pi - torch.pi
        )

        # Initialize the device
        dev = qml.device("default.qubit", wires=num_qubits)

        # Define the quantum circuit
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def gen_circuit(w):
            # random noise as generator input
            z1 = random.uniform(-1, 1)
            z2 = random.uniform(-1, 1)
            # construct generator circuit for both atom vector and node matrix
            for i in range(num_qubits):
                qml.RY(np.arcsin(z1), wires=i)
                qml.RZ(np.arcsin(z2), wires=i)
            for l in range(num_layers):
                for i in range(num_qubits):
                    qml.RY(w[l][i], wires=i)
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(w[l][i + num_qubits], wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self.gen_circuit = gen_circuit

    def forward(self, batch_size: int):
        sample_list = [
            torch.concat(
                [tensor.unsqueeze(0) for tensor in self.gen_circuit(self.weights)]
            )
            for _ in range(batch_size)
        ]
        noise = torch.stack(tuple(sample_list)).float()
        return noise


class QuantumShadowNoise(nn.Module):
    @staticmethod
    def build_qnode(num_qubits, num_layers, num_basis):

        paulis = [qml.PauliZ, qml.PauliX, qml.PauliY, qml.Identity]
        basis = [
            qml.operation.Tensor(*[random.choice(paulis)(i) for i in range(num_qubits)])
            for _ in range(num_basis)
        ]

        dev = qml.device("default.qubit", wires=num_qubits, shots=300)

        @qml.qnode(dev, interface="torch", diff_method="best")
        def gen_circuit(w):
            z1 = random.uniform(-1, 1)
            z2 = random.uniform(-1, 1)
            # construct generator circuit for both atom vector and node matrix
            for i in range(num_qubits):
                qml.RY(np.arcsin(z1), wires=i)
                qml.RZ(np.arcsin(z2), wires=i)
            for l in range(num_layers):
                for i in range(num_qubits):
                    qml.RY(w[l][i], wires=i)
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(w[l][i + num_qubits], wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])
            return qml.shadow_expval(basis)

        return basis, gen_circuit

    def __init__(
        self,
        z_dim: int,
        *,
        num_qubits: int = 8,
        num_layers: int = 3,
        num_basis: int = 3,
    ):
        super(QuantumShadowNoise, self).__init__()

        # Register the parameters with the module
        self.z_dim = z_dim
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_basis = num_basis

        self.basis, self.gen_circuit = self.build_qnode(
            num_qubits, num_layers, num_basis
        )

        # Initialize weights with PyTorch (between -pi and pi) and register as a learnable parameter
        self.weights = nn.Parameter(
            torch.rand(num_layers, (num_qubits * 2 - 1)) * 2 * torch.pi - torch.pi
        )
        self.coeffs = nn.Parameter(torch.rand(num_basis, self.z_dim))

    def forward(self, batch_size: int):
        sample_list = [
            torch.cat(
                [tensor.unsqueeze(0) for tensor in self.gen_circuit(self.weights)]
            )
            for _ in range(batch_size)
        ]
        noise = torch.stack(tuple(sample_list)).float()
        noise = torch.matmul(noise, self.coeffs)
        return noise


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
        return torch.rand(batch_size, self.z_dim).to(next(self.parameters()).device)

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


class QuantumGenerator(Generator):
    """Quantum Generator network of MolGAN"""

    def __init__(
        self,
        dataset,
        *,
        conv_dims=[128, 256, 512],
        z_dim=8,
        dropout=0.0,
        use_shadows=False,
    ):
        super(QuantumGenerator, self).__init__(
            dataset,
            conv_dims=conv_dims,
            z_dim=z_dim,
            dropout=dropout,
        )
        if use_shadows:
            self.noise_generator = QuantumShadowNoise(z_dim)
        else:
            self.noise_generator = QuantumMolGanNoise(z_dim)

    def _generate_z(self, batch_size):
        return self.noise_generator(batch_size)


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
