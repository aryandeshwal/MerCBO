import itertools
import numpy as np

import torch
from MerCBO.experiments.test_functions.experiment_configuration import ISING_GRID_H, ISING_GRID_W, \
    ISING_N_EDGES, CONTAMINATION_N_STAGES
from MerCBO.experiments.test_functions.experiment_configuration import sample_init_points, \
    generate_ising_interaction, generate_contamination_dynamics


LABS_DIM = 40 
class LABS_OBJ(object):
    """
    Low auto-correlation binaary 
    """
    def __init__(self, random_seed_pair=(None, None)):
        self.n_vertices = np.array([2] * LABS_DIM)
        self.suggested_init = torch.empty(0).long()
        self.suggested_init = torch.cat([self.suggested_init, sample_init_points(self.n_vertices, 20 - self.suggested_init.size(0), random_seed_pair[1])], dim=0)
        self.adjacency_mat = []
        self.fourier_freq = []
        self.fourier_basis = []
        self.random_seed_info = 'R'.join([str(random_seed_pair[i]).zfill(4) if random_seed_pair[i] is not None else 'None' for i in range(2)])
        for i in range(len(self.n_vertices)):
            n_v = self.n_vertices[i]
            adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
            self.adjacency_mat.append(adjmat)
            laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
            eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
            self.fourier_freq.append(eigval)
            self.fourier_basis.append(eigvec)

    def evaluate(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        assert x.size(1) == len(self.n_vertices)
        return torch.tensor([self._evaluate_single(x[i]) for i in range(x.size(0))]).float()

    def _evaluate_single(self, x):
        assert x.dim() == 1
        assert x.numel() == len(self.n_vertices)
        if x.dim() == 2:
            x = x.squeeze(0)
        x = x.numpy()
        N = x.shape[0] #x[x == 0] = -1
        #print(N)
        E = 0# energy 
        for k in range(1, N):
            C_k = 0
            for j in range(0, N-k-1):
                C_k += (-1)**(1-x[j]*x[j+k])
            E += (C_k**2)
        if (E==0):
            print('found zero')
        #print(-1*N/(2*E))
        return (-1. * N / (2*E))

