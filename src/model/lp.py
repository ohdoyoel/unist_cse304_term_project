import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

def label_propagation(A, labels, mask, alpha=0.6, max_iter=10, tol=1e-6):
    n = labels.size(0)
    Y = torch.zeros((n, labels.max().item() + 1), device=A.device)
    Y[mask, labels[mask]] = 1  # Initialize Y with one-hot encoding for labeled nodes

    for _ in range(max_iter):
        Y_new = alpha * torch.mm(A, Y) + (1 - alpha) * Y
        if torch.norm(Y_new - Y, p='fro') < tol:
            break
        Y = Y_new

    return Y.argmax(dim=1)
