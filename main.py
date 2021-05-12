from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

from braket.ocean_plugin import BraketSampler, BraketDWaveSampler
from dimod.binary_quadratic_model import BinaryQuadraticModel
from dwave.system.composites import EmbeddingComposite
from dwave.system import DWaveSampler
import dwave_networkx as dnx
import networkx as nx
import dwave_networkx as dnx
from collections import defaultdict

from dimod.reference.samplers import ExactSolver


class BaseChannel(ABC):
    def __init__(self, params, token_file="dwave-token.txt"):
        self.params = params
        # self.graph = graph
        # self.list_edges = np.array(graph.edges)

        with open(token_file) as f:
            token = str(f.read()).rstrip()
        self.sampler = DWaveSampler({'topology__type': 'chimera'}, token=token)
        self.list_edges = np.array(self.sampler.edgelist)
        self.list_nodes = np.array(self.sampler.nodelist)

        # self.sampler = ExactSolver()

        super().__init__()

    @abstractmethod
    def send(self, J_in, h_in):
        pass

    @abstractmethod
    def get_nishimori_temperature(self):
        pass

    @abstractmethod
    def conditional_density(self, y_out, y_in):
        pass

    def encode(self, x_in):
        x_dict = {node: x_in[i] for i, node in enumerate(self.list_nodes)}

        J = defaultdict(int)
        for (u,v) in self.list_edges:
            J[(u,v)] = - x_dict[u] * x_dict[v]
        h = {node: -x_dict[node] for node in x_dict.keys()}

        return J, h

    def decode(self, J_out, h_out, T=0):
        x_dec = np.array(list(self.sampler.sample_ising(h_out, J_out).first.sample.values()))
        
        return x_dec

    def get_ber(self, x_in, x_dec):
        N = len(x_in)
        return np.sum(np.abs(x_in - x_dec)) / (2*N)


class BinarySymmetricChannel(BaseChannel):
    def __init__(self, p_error):
        params = {"p_error": p_error}

        super().__init__(params)

    def send(self, J_in, h_in):
        p = self.params['p_error']
        J_out = {edge: J_in[edge] * np.random.choice([-1, 1], p=[p, 1-p]) for edge in J_in.keys()}
        h_out = {edge: h_in[edge] * np.random.choice([-1, 1], p=[p, 1-p]) for edge in h_in.keys()}
        
        return J_out, h_out

    def get_nishimori_temperature(self):
        p = self.params['p_error']
        return 2 / np.log((1-p)/p)

    def conditional_density(self, y_out, y_in):
        if y_out != y_in:
            return self.params['p_error']
        else:
            return 1-self.params['p_error']


if __name__ == "__main__":
    n_p = 10
    n_reps = 10

    channel = BinarySymmetricChannel(0.001)
    x_in = np.array(np.ones(len(channel.list_nodes)))

    # ber = np.zeros((n_p, n_reps))
    # list_p = np.linspace(0.01, 0.499, n_p)
    # for i_p, p_error in enumerate(list_p):
    #     print(f"================= p: {p_error} ==================")
    #     channel = BinarySymmetricChannel(p_error)

    #     for i_rep in range(n_reps):
    #         J_in, h_in = channel.encode(x_in)
    #         # print("y_in", y_in)
    #         J_out, h_out = channel.send(J_in, h_in)
    #         # print("y_out", y_out)
    #         x_dec = channel.decode(J_out, h_out)
    #         print("x_dec", x_dec)
    #         ber[i_p, i_rep] = channel.get_ber(x_in, list(x_dec))
    #         print(ber[i_p, i_rep])
    
    # plt.xlabel("Crossover probability")
    # plt.ylabel("BER (T=0)")
    # plt.plot(list_p, np.mean(ber, axis=1))
    # plt.show()
