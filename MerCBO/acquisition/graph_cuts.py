# three main steps
# define the relaxed objective
# setup the graph
# compute minimum cut solution
from graph_tool.all import *
import numpy as np
class GraphCuts():
    def __init__(self, n_vars,  alpha):
        self.n_vars = n_vars
        self.alpha = alpha
        self.b = alpha[:n_vars] 
        self.upper_indices = np.triu_indices(self.n_vars, 1)
        self.A = np.zeros((self.n_vars, self.n_vars))
        self.A[self.upper_indices] = self.alpha[n_vars:]
        self.A_pos = np.copy(self.A)
        self.A_neg = np.copy(self.A)
        self.A_pos[self.A_pos <= 0] = 0 
        self.A_neg[self.A_neg >= 0] = 0 
        self.one_vector = np.ones(self.n_vars)

    def initialize_gamma(self, random_set):
        if random_set:  
            self.gamma = np.random.uniform(size=(self.n_vars, self.n_vars))
        else:
            self.gamma = np.outer(self.one_vector, self.one_vector)/2

    def update_gamma(self, previous_res, t):
        temp = np.outer(self.one_vector, self.one_vector) - np.outer(previous_res, self.one_vector) - \
                                                             np.outer(self.one_vector, previous_res)
        self.S_t = np.multiply(self.A_pos, temp)
        self.gamma = self.gamma - (self.S_t/np.sqrt(t+1))
        self.gamma[self.gamma < 0] = 0.
        self.gamma[self.gamma > 1] = 1.

    def create_relaxed_objective(self):
        # separating out the non-negative and non-positive alphas in the quadratic term
        # non-positive alphas are submodular -> direcly representable as graphs
        # non-negative alphas needs to be relaxed to a linear from 
        self.hadamard_product = np.multiply(self.A_pos, self.gamma)
        self.h_gamma = np.dot(self.one_vector, self.hadamard_product) + \
                             np.dot(self.hadamard_product, self.one_vector) + self.b
        self.extra_sub = np.dot(np.dot(self.one_vector, self.hadamard_product), self.one_vector)



    def get_solution_min_cut(self, random_set, boykov, max_t):
        # add n_vars + 2 (source and target) vertices to the graph
        #print("---------------------------------------------------------------------")
        t = max_t
        best_value = np.inf 
        best_input = []
        all_inputs = []
        all_vals = []
        self.initialize_gamma(random_set)
        count = 0
        for n_iter in range(t):
            self.create_relaxed_objective()
            self.graph = Graph()
            self.graph.add_vertex(self.n_vars + 2)
            self.source = 0
            self.target = self.n_vars + 1
            # add the capacity property
            cap = self.graph.new_edge_property("double")
            # add edges for first order terms
            for i in range(self.n_vars):
                if (self.h_gamma[i] > 0):
                    e = self.graph.add_edge(self.source, i+1)
                    cap[e] = self.h_gamma[i]
                    # print("edge: ",self.source, i+1, cap[e])
                elif (self.h_gamma[i] < 0):
                    e = self.graph.add_edge(i+1, self.target)
                    cap[e] = -1*self.h_gamma[i]
                    # print("edge: ", i+1, self.target, cap[e])

            # add edges for second order terms
            # add (v_i, v_j) -alpha_{ij} capacity
            # add (v_j, t) -alpha_{ij} capacity
            for i in range(self.n_vars):
                for j in range(i+1, self.n_vars):
                    if self.A_neg[i][j] != 0:
                        count += 1
                        e = self.graph.add_edge(j+1, self.target)
                        cap[e] = -1*self.A_neg[i][j]
                        # print("edge: ", j+1, self.target, cap[e])
                        e = self.graph.add_edge(i+1, j+1)
                        cap[e] = -1*self.A_neg[i][j]
                        # print("edge: ", i+1, j+1, cap[e])
            # res = boykov_kolmogorov_max_flow(self.graph, self.source, self.target, cap)
            # print("partition", min_st_cut(self.graph, self.source, cap, res).get_array()[1:-1])

            # res.a = cap.a - res.a  # the actual flow
            # max_flow = sum(res[e] for e in self.graph.vertex(self.target).in_edges())
            # print("max flow boykov-kolmogorov: ", max_flow)
            if boykov:
                res = boykov_kolmogorov_max_flow(self.graph, self.source, self.target, cap)
            else:
                res = push_relabel_max_flow(self.graph, self.source, self.target, cap)
            # print("partition", min_st_cut(self.graph, self.source, cap, res).get_array()[1:-1])
            source_based_partition = min_st_cut(self.graph, self.source, cap, res).get_array()[1:-1].astype(np.float)
            partition = np.asarray([not(x) for x in source_based_partition])
            # res.a = cap.a - res.a  # the actual flow
            # max_flow = sum(res[e] for e in self.graph.vertex(self.target).in_edges())
            # print("max flow push-relabel: ", max_flow)
            new_val = self.alpha[0] + np.dot(partition.T, np.dot(self.A, partition)) + np.dot(self.b, partition)
            # print("actual value: ", self.alpha[0] + np.dot(partition.T, np.dot(self.A_neg, partition)) + \
            #                           np.dot(self.h_gamma, partition) - self.extra_sub)
            #print("New value: ", new_val)
            if (new_val < best_value):
                best_value = new_val
                best_input = partition
            #self.logger.debug("new_val: %s"%new_val)
            #print("-- Best_value:", best_value)
            #print(best_input)
            all_inputs.append(partition)
            all_vals.append(new_val)
            if not random_set:
                self.update_gamma(partition, n_iter)
            else:
                self.initialize_gamma(True)
            #res = push_relabel_max_flow(self.graph, self.source, self.target, cap)
        #print(partition.get_array())
        # to check that the objective is correct lower bound, we need to subtract an extra term
        # print("max difference: ", all_vals[0] - np.min(all_vals[1:]))
        #print("---------------------------------------------------------------------")
        # extra_sub = np.dot(np.dot(self.one_vector, self.hadamard_product), self.one_vector)
        #return partition.get_array()[1:-1], extra_sub
        return best_input, self.extra_sub, np.asarray(all_inputs), np.asarray(all_vals)










