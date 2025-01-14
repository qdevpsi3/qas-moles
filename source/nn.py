import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GINConv, SAGEConv, global_mean_pool
from torchdrug.data import Molecule

from .layers import GraphAggregation, GraphConvolution, MultiDenseLayers


def extract_graphs_from_features(features, library="pyg"):
    node_features = features["X"]
    adjacency_matrix = features["A"]
    edge_indices = (
        torch.tensor(
            [
                (src, dst)
                for src, row in enumerate(adjacency_matrix)
                for dst in range(len(row))
                if adjacency_matrix[src, dst] > 0
            ],
            dtype=torch.long,
        )
        .t()
        .contiguous()
    )

    edge_features = torch.tensor(
        [adjacency_matrix[src, dst] for src, dst in edge_indices.t()],
        dtype=torch.long,
    )

    if library == "pyg":
        # PyTorch Geometric Data object
        graph = Data(
            x=node_features,
            edge_index=edge_indices,
            edge_attr=edge_features,
        )
    elif library == "torchdrug":
        # TorchDrug Molecule object
        graph = Molecule(
            edge_list=edge_indices.t(),
            atom_type=node_features[:, 0].long(),
        )
    else:
        raise ValueError(
            f"Unsupported library: {library}. Choose 'pyg' or 'torchdrug'."
        )
    return graph


class BaseNN(nn.Module, ABC):
    """Abstract base class for all neural network modules with a build method."""

    _built = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registery = {}

    def forward(self, *args, **kwargs):
        if not self._built:
            raise RuntimeError("Module must be built before forward pass.")
        return self._forward(*args, **kwargs)

    def build(self, *args, **kwargs):
        """Build the module layers based on given input/output sizes or parameters."""
        super().__init__()
        self._build(*args, **kwargs)
        self._built = True

    @classmethod
    def register(cls, name, **kwargs):
        """Register the class with the factory."""
        cls._registery[name] = cls(**kwargs)

    @abstractmethod
    def _forward(self, *args, **kwargs):
        """This is where the forward logic lives after build has been called."""
        pass

    @abstractmethod
    def _build(self, *args, **kwargs):
        """Build the module layers based on given input/output sizes or parameters."""
        pass


class BaseGenerator(BaseNN):
    """Abstract base class for generators."""

    @abstractmethod
    def _build(
        self,
        num_vertices: int,
        num_bond_types: int,
        num_atom_types: int,
    ):
        pass

    @abstractmethod
    def _forward(self, batch_size: int):
        pass


class BaseDiscriminator(BaseNN, ABC):
    """Abstract base class for discriminators."""

    @abstractmethod
    def _build(
        self,
        num_vertices: int,
        num_bond_types: int,
        num_atom_types: int,
    ):
        pass

    @abstractmethod
    def _forward(self, x, a):
        pass


class BasePredictor(BaseNN, ABC):
    """Abstract base class for discriminators."""

    @abstractmethod
    def _build(
        self,
        num_vertices: int,
        num_bond_types: int,
        num_atom_types: int,
        num_metrics: int,
    ):
        pass

    @abstractmethod
    def _forward(self, x, a):
        pass


########################################
# Generator Class
########################################


@dataclass(unsafe_hash=True)
class MolGANGenerator(BaseGenerator):
    z_dim: int = 8
    conv_dims: List[int] = (128, 256)
    dropout: float = 0.0

    def _build(self, num_vertices: int, num_bond_types: int, num_atom_types: int):
        self._num_vertices = num_vertices
        self._num_bond_types = num_bond_types
        self._num_atom_types = num_atom_types
        self.multi_dense_layers = MultiDenseLayers(
            self.z_dim, self.conv_dims, nn.Tanh(), dropout_rate=self.dropout
        )
        self.edges_layer = nn.Linear(
            self.conv_dims[-1],
            num_bond_types * num_vertices * num_vertices,
        )
        self.nodes_layer = nn.Linear(self.conv_dims[-1], num_vertices * num_atom_types)
        self.dropout_layer = nn.Dropout(self.dropout)

    def _generate_z(self, batch_size: int):
        return torch.rand(batch_size, self.z_dim).to(next(self.parameters()).device)

    def _forward(self, batch_size: int):
        z = self._generate_z(batch_size)
        output = self.multi_dense_layers(z)

        edges_logits = self.edges_layer(output).view(
            -1,
            self._num_bond_types,
            self._num_vertices,
            self._num_vertices,
        )
        edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2
        edges_logits = self.dropout_layer(edges_logits.permute(0, 2, 3, 1))

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropout_layer(
            nodes_logits.view(
                -1,
                self._num_vertices,
                self._num_atom_types,
            )
        )

        return edges_logits, nodes_logits


class QMolGANNoise(nn.Module):
    def __init__(self, num_qubits: int = 8, num_layers: int = 3):
        super(QMolGANNoise, self).__init__()

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


@dataclass(unsafe_hash=True)
class QMolGANGenerator(MolGANGenerator):
    num_circuit_qubits: int = 8
    num_circuit_layers: int = 3
    conv_dims: Tuple[int] = (128, 256)
    dropout: float = 0.0

    def _build(
        self,
        num_vertices: int,
        num_bond_types: int,
        num_atom_types: int,
    ):
        self.z_dim = self.num_circuit_qubits
        super()._build(num_vertices, num_bond_types, num_atom_types)
        self.q_noise = QMolGANNoise(self.num_circuit_qubits, self.num_circuit_layers)

    def _generate_z(self, batch_size: int):
        return self.noise_generator(batch_size)


########################################
# Discriminator Class
########################################
@dataclass(unsafe_hash=True)
class MolGANDiscriminator(BaseDiscriminator):
    """Discriminator network of MolGAN"""

    conv_dims: Tuple[List[int]] = ((128, 64), 128, (128, 64))
    f_dim: int = 0
    with_features: bool = False
    dropout: float = 0.0

    def _build(
        self,
        num_vertices: int,
        num_bond_types: int,
        num_atom_types: int,
        num_metrics: int,
    ):
        m_dim = num_atom_types
        b_dim = num_bond_types - 1
        graph_conv_dim, aux_dim, linear_dim = self.conv_dims

        self.activation_f = nn.Tanh()
        graph_conv_dim, aux_dim, linear_dim = self.conv_dims
        self.gcn_layer = GraphConvolution(
            m_dim,
            graph_conv_dim,
            b_dim,
            self.with_features,
            self.f_dim,
            self.dropout,
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

    def _forward(self, x, a):
        adj = a[:, :, :, 1:].permute(0, 3, 1, 2)
        h = self.gcn_layer(x, adj)
        h = self.agg_layer(x, h)
        h = self.multi_dense_layers(h)
        output = self.output_layer(h)
        return output, h


@dataclass(unsafe_hash=True)
class GNNDiscriminator(BaseDiscriminator):
    """GNN-based Discriminator for MolGAN."""

    conv_dim: int = 128
    dropout: float = 0.1

    def _build(self, num_vertices: int, num_bond_types: int, num_atom_types: int):
        self.num_vertices = num_vertices
        self.num_bond_types = num_bond_types
        self.num_atom_types = num_atom_types

        self.gnn_layer = GATConv(
            in_channels=self.num_atom_types,
            out_channels=self.conv_dim,
            edge_dim=self.num_bond_types,
        )
        self.output_layer = nn.Linear(self.conv_dim, 1)

    def _forward(self, adjacency_tensor, node_features, activation=None):
        # Convert adjacency and node features into PyTorch Geometric Batch objects
        batch_graphs = [
            extract_graphs_from_features({"A": a, "X": x})
            for a, x in zip(adjacency_tensor, node_features)
        ]
        batch_graphs = Batch.from_data_list(batch_graphs)

        # Pass node features and edges through the GNN layer
        batch_features = self.gnn_layer(
            batch_graphs.x, batch_graphs.edge_index, batch_graphs.edge_attr
        )

        # Aggregate graph-level features by averaging over nodes
        batch_features = batch_features.view(batch_graphs.num_graphs, -1, self.conv_dim)
        graph_level_features = batch_features.mean(dim=1)

        # Pass graph-level features through the output layer
        output = self.output_layer(graph_level_features)

        # Apply optional activation
        if activation is not None:
            output = activation(output)

        return output, graph_level_features


@dataclass(unsafe_hash=True)
class GINDiscriminator(BaseDiscriminator):
    """
    A GIN-based Discriminator for MolGAN-like architectures.
    """

    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1

    def _build(self, num_vertices: int, num_bond_types: int, num_atom_types: int):
        self.num_vertices = num_vertices
        self.num_bond_types = num_bond_types
        self.num_atom_types = num_atom_types

        # Build multiple GINConv layers
        self.gin_convs = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            # The input dimension for the first layer is num_atom_types;
            # afterwards, it's hidden_dim
            mlp_input_dim = self.num_atom_types if layer_idx == 0 else self.hidden_dim
            mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.gin_convs.append(GINConv(mlp))

        self.dropout_layer = nn.Dropout(self.dropout)
        self.output_layer = nn.Linear(self.hidden_dim, 1)

    def _forward(
        self,
        adjacency_tensor,
        node_features,
        activation: Optional[nn.Module] = None,
    ):
        # Convert adjacency and node features into PyTorch Geometric data Batch
        batch_graphs = [
            extract_graphs_from_features({"A": a, "X": x})
            for a, x in zip(adjacency_tensor, node_features)
        ]
        batch_graphs = Batch.from_data_list(batch_graphs)

        # x: node feature matrix; edge_index: adjacency list; edge_attr: edge features
        x, edge_index, edge_attr, batch_idx = (
            batch_graphs.x,
            batch_graphs.edge_index,
            batch_graphs.edge_attr,
            batch_graphs.batch,
        )

        # Pass through each GIN layer
        for gin_conv in self.gin_convs:
            x = gin_conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout_layer(x)

        # Global average pooling to get graph-level features
        graph_level_features = global_mean_pool(x, batch_idx)

        # Output layer
        logits = self.output_layer(graph_level_features)

        # Optional activation
        if activation is not None:
            logits = activation(logits)

        return logits, graph_level_features


@dataclass(unsafe_hash=True)
class GraphSAGEDiscriminator(BaseDiscriminator):
    """
    A GraphSAGE-based Discriminator for MolGAN-like architectures.
    """

    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1

    def _build(self, num_vertices: int, num_bond_types: int, num_atom_types: int):
        self.num_vertices = num_vertices
        self.num_bond_types = num_bond_types
        self.num_atom_types = num_atom_types

        self.convs = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            in_dim = self.num_atom_types if layer_idx == 0 else self.hidden_dim
            self.convs.append(SAGEConv(in_dim, self.hidden_dim))

        self.dropout_layer = nn.Dropout(self.dropout)
        self.output_layer = nn.Linear(self.hidden_dim, 1)

    def _forward(
        self, adjacency_tensor, node_features, activation: Optional[nn.Module] = None
    ):
        batch_graphs = [
            extract_graphs_from_features({"A": a, "X": x})
            for a, x in zip(adjacency_tensor, node_features)
        ]
        batch_graphs = Batch.from_data_list(batch_graphs)

        x, edge_index, edge_attr, batch_idx = (
            batch_graphs.x,
            batch_graphs.edge_index,
            batch_graphs.edge_attr,
            batch_graphs.batch,
        )

        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout_layer(x)

        graph_level_features = global_mean_pool(x, batch_idx)
        logits = self.output_layer(graph_level_features)
        if activation is not None:
            logits = activation(logits)

        return logits, graph_level_features
