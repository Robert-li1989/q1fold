"""
Q1Fold Quantum 1D Feature Extraction Layer

A 6-qubit variational quantum circuit that processes 3-nucleotide sliding windows
from RNA sequences for feature extraction.
"""

import warnings
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*", category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np


class QuantumConv1D(nn.Module):
    """
    Quantum Convolutional 1D Layer for RNA feature extraction.
    
    Uses a 6-qubit variational quantum circuit with:
    - Biochemically-motivated nucleotide encoding (q0-q2)
    - Position encoding (q3)
    - Context encoding (q4)
    - Purine-pyrimidine balance encoding (q5)
    """
    
    def __init__(self, seed=42):
        super().__init__()
        self.window_size = 3
        self.n_qubits = 6
        self.n_layers = 1
        
        # Nucleotide to angle mapping (biochemically motivated)
        self.nucleotide_angles = {
            'A': [0, 0],           # Purine, Adenine
            'G': [np.pi/2, 0],     # Purine, Guanine
            'C': [0, np.pi/2],     # Pyrimidine, Cytosine
            'U': [np.pi/2, np.pi/2] # Pyrimidine, Uracil
        }
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # Learnable parameters for variational layers
        self.theta = nn.Parameter(torch.empty(self.n_layers, self.n_qubits, 3))
        self.theta.data[:, :, :] = torch.randn(self.n_layers, self.n_qubits, 3) * 0.3

        # Learnable embedding weights
        self.embed_weights = nn.Parameter(torch.randn(4, 2) )  # 4 nucleotides, 2 angle params
        
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(weights, embed_weights, nucleotide_indices, pos_batch):
            """
            Enhanced quantum circuit with better nucleotide embedding
            
            Args:
                weights: Learnable rotation parameters [n_layers, n_qubits, 3]
                embed_weights: Learnable nucleotide embedding weights [4, 2]
                nucleotide_indices: One-hot encoded nucleotides [batch_size, 4, 3]
                pos_batch: Position encoding [batch_size]
            """
            
            # 1. Enhanced nucleotide embedding with learnable parameters
            for pos_in_window in range(3):
                base_qubit = pos_in_window  # Use first 3 qubits for nucleotides
                
                # Get one-hot encoded nucleotide for this position
                nt_onehot = nucleotide_indices[:, :, pos_in_window]  # [batch_size, 4]
                
                # Find which nucleotide is present (argmax of one-hot)
                nt_idx = torch.argmax(nt_onehot, dim=1)  # [batch_size] - indices 0,1,2,3 for A,G,C,U
                device = nt_idx.device
                
                # Base angles from biochemical properties (fixed per nucleotide type)
                base_angles = torch.zeros(4, 2, device=device)  # [4 nucleotides, 2 angles]
                base_angles[0] = torch.tensor([0, 0], device=device)           # A: Purine, Adenine
                base_angles[1] = torch.tensor([np.pi/2, 0], device=device)     # G: Purine, Guanine  
                base_angles[2] = torch.tensor([0, np.pi/2], device=device)     # C: Pyrimidine, Cytosine
                base_angles[3] = torch.tensor([np.pi/2, np.pi/2], device=device) # U: Pyrimidine, Uracil
                
                # Get angles for the specific nucleotides in this batch
                batch_base_angles = base_angles[nt_idx]  # [batch_size, 2]
                
                # Add learnable component (same for all samples in batch)
                batch_learned_angles = embed_weights[nt_idx]  # [batch_size, 2]
                
                # Final angles combining base + learned
                final_angles = batch_base_angles + batch_learned_angles  # [batch_size, 2]
                
                # Apply rotations - each sample gets its specific nucleotide's angles
                qml.RY(final_angles[:, 0], wires=base_qubit)
                qml.RZ(final_angles[:, 1], wires=base_qubit)
            
            # 2. Position encoding on dedicated qubits
            pos_qubit = 3
            qml.RY(pos_batch * np.pi/4, wires=pos_qubit)
            
            # 3. Context energy encoding (for sequence context)
            context_qubit = 4
            # Encode 3-mer context as combination of all positions
            context_weights = torch.tensor([1.1, 1.6, 1.6, 1.1], device=device)  # A, C, G, U weights
            context_angle = torch.sum(nucleotide_indices * context_weights.view(1, 4, 1), dim=[1,2]) * np.pi/4
            qml.RY(context_angle, wires=context_qubit)
            
            # 4. Additional auxiliary qubit for global features
            aux_qubit = 5
            
            # Purine vs Pyrimidine balance (has biochemical meaning)
            purine_mask = torch.tensor([1.0, 1.0, 0.0, 0.0], device=device)  # A,G = purines
            pyrimidine_mask = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device)  # C,U = pyrimidines

            purine_count = torch.sum(nucleotide_indices * purine_mask.view(1, 4, 1), dim=[1,2])
            pyrimidine_count = torch.sum(nucleotide_indices * pyrimidine_mask.view(1, 4, 1), dim=[1,2])
            purine_pyrimidine_balance = (purine_count - pyrimidine_count) * np.pi/4

            qml.RX(purine_pyrimidine_balance, wires=aux_qubit)
            
            # 5. Variational layers with improved connectivity
            for layer_idx in range(weights.shape[0]):
                # Single-qubit rotations
                for q in range(self.n_qubits):
                    p, t, o = weights[layer_idx, q, 0], weights[layer_idx, q, 1], weights[layer_idx, q, 2]
                    qml.Rot(p, t, o, wires=q)
                
                # Improved entangling structure - Linear chain
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])  # Close the ring
            
            # 6. Comprehensive measurement strategy
            measurements = []
            
            # Single-qubit measurements (all Pauli operators)
            for q in range(self.n_qubits):
                measurements.extend([
                    qml.expval(qml.PauliX(q)),
                    qml.expval(qml.PauliY(q)),
                    qml.expval(qml.PauliZ(q))
                ])
            
            # Two-qubit correlations (focus on meaningful pairs)
            important_pairs = [
                (0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3), (3, 4), (4, 5), (0, 5), (1, 5), (2, 5)
            ]

            for q1, q2 in important_pairs:
                measurements.extend([
                    qml.expval(qml.PauliZ(q1) @ qml.PauliZ(q2)),
                    qml.expval(qml.PauliX(q1) @ qml.PauliX(q2)),
                    qml.expval(qml.PauliY(q1) @ qml.PauliY(q2))
                ])
            
            # Add triple correlations
            triple_correlations = [
                (0, 1, 2),  # All nucleotides
            ]

            for q1, q2, q3 in triple_correlations:
                measurements.append(qml.expval(qml.PauliZ(q1) @ qml.PauliZ(q2) @ qml.PauliZ(q3)))
            
            return measurements
        
        self.circuit = circuit
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor [B, 4, L] - one-hot encoded nucleotide sequences
        Returns:
            Quantum features [B, n_features, L]
        """
        B, _, L = x.shape
        
        # Create sliding windows
        x_padded = F.pad(x, (1, 1))
        windows = x_padded.unfold(2, self.window_size, 1)  # [B, 4, L, 3]
        windows = windows.permute(0, 2, 1, 3).reshape(-1, 4, 3)  # [B*L, 4, 3]
        
        # Position encoding
        pos_codes = torch.linspace(0, 1, L, device=x.device)
        pos_batch = pos_codes.repeat_interleave(B)
        
        # Apply quantum circuit
        out = self.circuit(self.theta, self.embed_weights, windows, pos_batch)
        
        # Process output
        if isinstance(out, (list, tuple)):
            out = torch.stack(out, dim=0).transpose(0, 1)
        elif out.ndim == 2:
            out = out.transpose(0, 1)

        total_features = 52  # 18 single-qubit + 33 two-qubit + 1 triple
        
        # Reshape: [B*L, n_features] -> [B, L, n_features] -> [B, n_features, L]
        out = out.view(B, L, total_features).permute(0, 2, 1).contiguous()
        
        return out.float()
