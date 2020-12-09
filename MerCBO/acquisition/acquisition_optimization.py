import os
import sys
import time
import psutil
from scipy.special import comb

import numpy as np

import torch
import torch.multiprocessing as mp

from MerCBO.acquisition.acquisition_optimizers.starting_points import optim_inits
from MerCBO.acquisition.acquisition_optimizers.greedy_ascent import greedy_ascent
from MerCBO.acquisition.acquisition_optimizers.simulated_annealing import simulated_annealing
from MerCBO.acquisition.acquisition_functions import expected_improvement
from MerCBO.acquisition.acquisition_marginalization import prediction_statistic
from MerCBO.acquisition.acquisition_optimizers.graph_utils import neighbors
from sympy.utilities.iterables import multiset_permutations
from MerCBO.graphGP.inference.inference import Inference
from MerCBO.graphGP.sampler.tool_partition import group_input
from MerCBO.acquisition import graph_cuts 

MAX_N_ASCENT = float('inf')
N_CPU = os.cpu_count()
N_AVAILABLE_CORE = min(10, N_CPU)
N_SA_RUN = 10

import cvxpy as cvx
from itertools import combinations


def compute_mercer_features(x, log_beta, one_set, double_set):
    if (len(x.shape) == 1):
        x = x[np.newaxis, :]
    beta = np.array(np.exp(log_beta.numpy()))
    phi_x = np.sqrt(np.exp(-2*beta[np.argwhere(one_set)[:, 1]])) * ((-1)**(np.dot(x, one_set)))
    double_idxs = np.argwhere(double_set)
    phi_x = np.concatenate([phi_x, np.sqrt(np.exp(-2*(beta[double_idxs[range(0, 2*len(double_set), 2), 1]] + beta[double_idxs[range(1, 2*len(double_set), 2), 1]]))) * ((-1)**(np.dot(x, double_set.T)))], axis = 1)
    return phi_x



def next_evaluation_with_thompson_sampling(input_data, output_data, inference_samples, partition_samples, log_noise_var, n_vertices, log_beta):
    print('(%s) Acquisition function (THOMPSON SAMPLING) optimization initial points selection began' % (time.strftime('%H:%M:%S', time.localtime(time.time()))))
    noise_var = np.exp(log_noise_var.numpy())
    val_set = np.zeros(input_data.size(1))
    # generate all permutations of order 1
    val_set[0] = 1
    one_set = np.array(list(multiset_permutations(val_set)))
    # generate all permutations of order 2
    start_time = time.time()
    val_set[1] = 1
    double_set = np.array(list(multiset_permutations(val_set)))
    one_set = one_set[::-1]
    double_set = double_set[::-1]
    phi = compute_mercer_features(input_data, log_beta, one_set, double_set)
    print("phi size: ", phi.shape)
    X, _, _, y, muY = standardise(phi, output_data.numpy())
    beta =  np.array(np.exp(log_beta.numpy())) 
    beta = np.concatenate([beta, (np.outer(beta, beta))[np.triu_indices(n_vertices.shape[0], 1)]])
    theta = fastmvg(X/np.sqrt(noise_var), y.squeeze(-1)/np.sqrt(noise_var), noise_var*np.diag(beta**2))

    # Submodular relaxation based AFO 
    beta = np.exp(log_beta.numpy())
    upper_indices = np.triu_indices(n_vertices.shape[0], 1)
    A = np.zeros((n_vertices.shape[0], n_vertices.shape[0]))
    A[upper_indices] = theta[n_vertices.shape[0]:] *  ((np.outer(beta, beta))[upper_indices])
    A.T[upper_indices] = theta[n_vertices.shape[0]:] * ((np.outer(beta, beta))[upper_indices])
    sdp_objective = theta.copy()
    sdp_objective[:n_vertices.shape[0]] *= np.sqrt(np.exp(-2*beta[np.argwhere(one_set)[:, 1]]))
    sdp_objective[:n_vertices.shape[0]] += np.sum(A, axis=0) #* np.sum(np.sqrt(np.exp(-2*beta)))
    sdp_objective[:n_vertices.shape[0]] = -2*sdp_objective[:n_vertices.shape[0]]
    double_idxs = np.argwhere(double_set)
    sdp_objective[n_vertices.shape[0]:] *= (4*np.sqrt(np.exp(-2*(beta[double_idxs[range(0, 2*len(double_set), 2), 1]] + beta[double_idxs[range(1, 2*len(double_set), 2), 1]]))))
    start_time = time.time()
    graphobj = graph_cuts.GraphCuts(n_vertices.shape[0], sdp_objective)
    print(f"time to optimize via Submodular Relaxation based AFO: {time.time() - start_time}") 
    random_set = False
    x_gc, extra_sub, all_points, all_vals = graphobj.get_solution_min_cut(random_set, False, 10)
    x_new = torch.from_numpy(x_gc.astype(np.long))
    print("Submodular Relaxation based AFO solution")
    print(x_new)

    if (torch.all(x_new == input_data, dim = 1).any()):
        x_new = torch.from_numpy(np.random.randint(low=0, high=2, size=(n_vertices.shape[0])))
        print("returning random")
        print("----------------------------------------------------")

    mean, std, var = prediction_statistic(x_new, inference_samples, partition_samples, n_vertices)
    return x_new, mean, std, var#, SA_model

def standardise(X, y):
    # Standardize the covariates to have zero mean and x_i'x_i = 1
    # set params
    n = X.shape[0]
    meanX = np.mean(X, axis=0)
    stdX  = np.std(X, axis=0) * np.sqrt(n)

    # Standardize y's
    meany = np.mean(y)
    y = y - meany

    return (X, meanX, stdX, y, meany)


def fastmvg(Phi, alpha, D):
    # Fast sampler for multivariate Gaussian distributions (large p, p > n) of 
    #  the form N(mu, S), where
    #       mu = S Phi' y
    #       S  = inv(Phi'Phi + inv(D))
    # Reference: 
    #   Fast sampling with Gaussian scale-mixture priors in high-dimensional 
    #   regression, A. Bhattacharya, A. Chakraborty and B. K. Mallick
    #   arXiv:1506.04778
    # print("Phi ", Phi.shape)
    # print("alpha ", alpha.shape)
    # print("D ", D.shape)
    n, p = Phi.shape

    d = np.diag(D)
    u = np.random.randn(p) * np.sqrt(d)
    delta = np.random.randn(n)
    v = np.dot(Phi,u) + delta
    mult_vector = np.vectorize(np.multiply)
    Dpt = mult_vector(Phi.T, d[:,np.newaxis])
    w = np.linalg.solve(np.matmul(Phi,Dpt) + np.eye(n), alpha - v)
    x = u + np.dot(Dpt,w)

    return x


# COMBO's AFO procedure below

def next_evaluation(x_opt, input_data, inference_samples, partition_samples, edge_mat_samples, n_vertices,
                    acquisition_func=expected_improvement, reference=None, parallel=None):
    """
    In case of '[Errno 24] Too many open files', check 'nofile' limit by typing 'ulimit -n' in a terminal
    if it is too small then add lines to '/etc/security/limits.conf'
        *               soft    nofile          [Large Number e.g 65536]
        *               soft    nofile          [Large Number e.g 65536]
    Rebooting may be needed.
    :param x_opt: 1D Tensor
    :param input_data:
    :param inference_samples:
    :param partition_samples:
    :param edge_mat_samples:
    :param n_vertices: 1d np.array
    :param acquisition_func:
    :param reference: numeric(float)
    :param parallel:
    :return:
    """
    id_digit = np.ceil(np.log(np.prod(n_vertices)) / np.log(10))
    id_unit = torch.from_numpy(np.cumprod(np.concatenate([np.ones(1), n_vertices[:-1]])).astype(np.int))
    fmt_str = '\t %5.2f (id:%' + str(id_digit) + 'd) ==> %5.2f (id:%' + str(id_digit) + 'd)'

    start_time = time.time()
    print('(%s) Acquisition function (EI) optimization initial points selection began'
          % (time.strftime('%H:%M:%S', time.localtime(start_time))))

    x_inits, acq_inits = optim_inits(x_opt, inference_samples, partition_samples, edge_mat_samples, n_vertices,
                                     acquisition_func, reference)
    n_inits = x_inits.size(0)
    assert n_inits % 2 == 0

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('(%s) Acquisition function optimization initial points selection ended - %s'
          % (time.strftime('%H:%M:%S', time.localtime(end_time)), time.strftime('%H:%M:%S', time.localtime(elapsed_time))))

    start_time = time.time()
    print('(%s) Acquisition function optimization with %2d inits'
          % (time.strftime('%H:%M:%S', time.localtime(start_time)), x_inits.size(0)))

    ga_args_list = [(x_inits[i], inference_samples, partition_samples, edge_mat_samples,
                     n_vertices, acquisition_func, MAX_N_ASCENT, reference) for i in range(n_inits)]
    ga_start_time = time.time()
    sys.stdout.write('    Greedy Ascent  began at %s ' % time.strftime('%H:%M:%S', time.localtime(ga_start_time)))
    if parallel:
        with mp.Pool(processes=min(n_inits, N_CPU // 3)) as pool:
            ga_result = []
            process_started = [False] * n_inits
            process_running = [False] * n_inits
            process_index = 0
            while process_started.count(False) > 0:
                cpu_usage = psutil.cpu_percent(0.25)
                run_more = (100.0 - cpu_usage) * float(psutil.cpu_count()) > 100.0 * N_AVAILABLE_CORE
                if run_more:
                    ga_result.append(pool.apply_async(greedy_ascent, args=ga_args_list[process_index]))
                    process_started[process_index] = True
                    process_running[process_index] = True
                    process_index += 1
            while [not res.ready() for res in ga_result].count(True) > 0:
                time.sleep(1)
            ga_return_values = [res.get() for res in ga_result]
    else:
        ga_return_values = [greedy_ascent(*(ga_args_list[i])) for i in range(n_inits)]
    ga_opt_vrt, ga_opt_acq = zip(*ga_return_values)
    sys.stdout.write('and took %s\n' % time.strftime('%H:%M:%S', time.localtime(time.time() - ga_start_time)))
    print('  '.join(['%.3E' % ga_opt_acq[i] for i in range(n_inits)]))

    opt_vrt = list(ga_opt_vrt[:])
    opt_acq = list(ga_opt_acq[:])

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('(%s) Acquisition function optimization ended %s'
          % (time.strftime('%H:%M:%S', time.localtime(end_time)), time.strftime('%H:%M:%S', time.localtime(elapsed_time))))

    # argsort sorts in ascending order so it is negated to have descending order
    acq_sort_inds = np.argsort(-np.array(opt_acq))
    suggestion = None
    for i in range(len(opt_vrt)):
        ind = acq_sort_inds[i]
        if not torch.all(opt_vrt[ind] == input_data, dim=1).any():
            suggestion = opt_vrt[ind]
            break
    if suggestion is None:
        for i in range(len(opt_vrt)):
            ind = acq_sort_inds[i]
            nbds = neighbors(opt_vrt[ind], partition_samples, edge_mat_samples, n_vertices, uniquely=True)
            for j in range(nbds.size(0)):
                if not torch.all(nbds[j] == input_data, dim=1).any():
                    suggestion = nbds[j]
                    break
            if suggestion is not None:
                break
    if suggestion is None:
        suggestion = torch.cat(tuple([torch.randint(low=0, high=int(n_v), size=(1, 1)) for n_v in n_vertices]), dim=1).long()

    mean, std, var = prediction_statistic(suggestion, inference_samples, partition_samples, n_vertices)
    return suggestion, mean, std, var


