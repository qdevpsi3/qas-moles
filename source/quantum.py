import random

import numpy as np
import pennylane as qml
import torch
from torch import nn


class NoiseQuantumGenerator(nn.Module):
    def __init__(self, num_qubits: int = 8, num_layers: int = 3):
        super(NoiseQuantumGenerator, self).__init__()

        # Register the parameters with the module
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        # Initialize weights with PyTorch (between -pi and pi) and register as a learnable parameter
        self.weights = nn.Parameter(
            torch.rand(num_layers * (num_qubits * 2 - 1)) * 2 * torch.pi - torch.pi
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
                    qml.RY(w[i], wires=i)
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(w[i + num_qubits], wires=i + 1)
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


# Quantum circuit configuration
n_qubits = 4  # Total number of qubits
n_a_qubits = 1  # Number of ancillary qubits
q_depth = 6  # Depth of the quantum circuit
patch_multiplier = 1  # Multiplier for patch measurement
bond_matrix_size = 9  # Bond matrix size
upper_triangle_number = (bond_matrix_size * bond_matrix_size - bond_matrix_size) // 2
output_size_subGen = 5  # Size of each sub-generator output

# Quantum simulator device (can switch to lightning.qubit if needed)
dev = qml.device("default.qubit", wires=n_qubits)

# Set device for Torch (CPU or CUDA if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Quantum circuit using PennyLane and Torch
@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(noise, weights):
    """Define a quantum circuit with parameterized layers."""
    weights = weights.reshape(q_depth, n_qubits)

    # Initialize qubits with noise (latent vectors)
    for i in range(n_qubits):
        qml.RY(noise[i], wires=i)

    # Apply layers of the circuit
    for i in range(q_depth):
        for y in range(n_qubits):
            qml.RY(weights[i][y], wires=y)
        for y in range(n_qubits - 1):
            qml.CZ(wires=[y, y + 1])

    return qml.probs(wires=list(range(n_qubits)))


def partial_measure(noise, weights, p_size):
    """Performs a partial measurement on the quantum circuit output."""
    probs = quantum_circuit(noise, weights)
    probsgiven5 = probs[:p_size]
    probsgiven5 /= torch.sum(probs)
    return torch.nn.functional.softmax(probsgiven5, -1).float().unsqueeze(0)


def partial_measure_3(noise, weights, p_size):
    """Performs a partial measurement on the quantum circuit output for the patch method."""
    probs = quantum_circuit(noise, weights)
    probsgiven15 = probs[: p_size * 3]
    probsgiven15 /= torch.sum(probs)
    q = torch.nn.functional.softmax(probsgiven15, -1)
    return torch.cat(
        [
            q[:p_size].float().unsqueeze(0) * patch_multiplier,
            q[p_size : p_size * 2].float().unsqueeze(0) * patch_multiplier,
            q[p_size * 2 : p_size * 3].float().unsqueeze(0) * patch_multiplier,
        ],
        0,
    )


class PatchQuantumGenerator(nn.Module):
    """Quantum generator class for the patch method."""

    def __init__(self, num_generators, q_delta=1):
        """
        Args:
            num_generators (int): Number of sub-generators to be used.
            q_delta (float, optional): Spread of the random distribution for parameter initialization.
        """
        super().__init__()
        tensor_mean = torch.zeros(q_depth, n_qubits, dtype=torch.float32)
        tensor_std = torch.full((q_depth, n_qubits), 1.05, dtype=torch.float32)

        # Initialize quantum parameters with random normal distribution
        self.q_params = nn.ParameterList(
            [
                nn.Parameter(
                    q_delta * torch.normal(mean=tensor_mean, std=tensor_std),
                    requires_grad=True,
                )
                for _ in range(num_generators // patch_multiplier)
            ]
        )
        self.num_generators = num_generators

    def forward(self, x):
        """Forward pass of the quantum generator."""
        patch_size = output_size_subGen
        edges = torch.Tensor(x.size(0), 0).to(device)
        nodes = torch.Tensor(x.size(0), 0).to(device)
        a = torch.triu_indices(bond_matrix_size, bond_matrix_size, offset=1)

        if patch_multiplier not in [1, 3]:
            raise ValueError(
                "Patch measurement undefined for patch_multiplier != 1 or 3"
            )

        # Generate edges and nodes for each input in the batch
        for jj, elem in enumerate(x):
            patches_edges_list = []
            patches_nodes = torch.Tensor(0, patch_size).to(device)

            for ii, params in enumerate(self.q_params):
                q_out = (
                    partial_measure(elem, params, patch_size)
                    if patch_multiplier == 1
                    else partial_measure_3(elem, params, patch_size)
                )

                if ii < upper_triangle_number // patch_multiplier:
                    patches_edges_list.append(q_out)
                else:
                    patches_nodes = torch.cat((patches_nodes, q_out))

            edge = torch.empty(bond_matrix_size, bond_matrix_size, patch_size).to(
                device
            )
            for ii, q in enumerate(torch.cat(tuple(patches_edges_list))):
                row, col = a[0][ii], a[1][ii]
                edge[row, col, :] = q
                edge[col, row, :] = q

            node = torch.reshape(patches_nodes, (bond_matrix_size, patch_size))

            if jj == 0:
                edges = edge.unsqueeze(0)
                nodes = node.unsqueeze(0)
            else:
                edges = torch.cat((edges, edge.unsqueeze(0)), 0)
                nodes = torch.cat((nodes, node.unsqueeze(0)), 0)

        return edges, nodes
