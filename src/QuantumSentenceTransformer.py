import pandas as pd
import numpy as np

import torch
import torch.nn as nn

import pennylane as qml


from sentence_transformers import SentenceTransformer



###################### CONFIG ############################

n_qubits = 4                # Number of qubits
step = 0.001               # Learning rate
batch_size = 32              # Number of samples for each training step
num_epochs = 30              # Number of training epochs
q_depth = 6                 # Depth of the quantum circuit (number of variational layers)
gamma_lr_scheduler = 0.1    # Learning rate reduction applied every 10 epochs.
q_delta = 0.01              # Initial spread of random quantum weights
start_time = time.time()    # Start of the computation timer

#########################################################

dev = qml.device("default.qubit", wires=n_qubits)

def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates.
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis.
    """
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOT.
    """
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])
        
        
@qml.qnode(dev, interface="torch")
def quantum_net(q_input_features, q_weights_flat):
    """
    The variational quantum circuit.
    """

    # Reshape weights
    q_weights = q_weights_flat.reshape(q_depth, n_qubits)

    # Start from state |+> , unbiased w.r.t. |0> and |1>
    H_layer(n_qubits)

    # Embed features in the quantum node
    RY_layer(q_input_features)

    # Sequence of trainable variational layers
    for k in range(q_depth):
        entangling_layer(n_qubits)
        RY_layer(q_weights[k])

    # Expectation values in the Z basis
    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
    return tuple(exp_vals)



class QuantumSentenceTransformer(nn.Module):
    """
    Torch module implementing the *dressed* quantum net.
    """

    def __init__(self, 
                 freeze_bert_layers = "all",       
                 pretrained_model = "distiluse-base-multilingual-cased-v1", 
                 device='cpu'):
        
        """
        Definition of the *dressed* layout.
        """

        super().__init__()
        
        # init Sentence Transformer and freeze all its layers by default. Only Quantum Layers are going
        # to be trainable
        self.device = device
        self.sentence_transformer  =  SentenceTransformer(pretrained_model, device=device)
        for param in self.sentence_transformer.parameters():
            if freeze_bert_layers == "all" or param in freeze_bert_layers:
                param.requires_grad = False
            
        self.pre_net = nn.Linear(512, n_qubits)
        self.q_params = nn.Parameter(q_delta * torch.randn(q_depth * n_qubits))
        self.post_net = nn.Linear(n_qubits, 2)

    def forward(self, input_text):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """
        
        #generate embeddings from input text
        input_features = self.sentence_transformer.encode(input_text, convert_to_tensor=True)
        # obtain the input features from the text embeddings for the quantum circuit
        # by reducing the feature dimension from 512 to 4
        #KIKIU
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0 #TEST ReLU

        # Apply the quantum circuit to each element of the batch and append to q_out
        q_out = torch.Tensor(0, n_qubits)
        q_out = q_out.to(self.device)
        for elem in q_in:
            q_out_elem = quantum_net(elem, self.q_params).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))

        # return the two-dimensional prediction from the postprocessing layer
        return self.post_net(q_out)