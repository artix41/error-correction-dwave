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


# def get_ground_state(hamiltonian):
#     braket_sampler = BraketDWaveSampler(s3,'arn:aws:braket:::device/qpu/d-wave/Advantage_system1')
#     sampler = EmbeddingComposite(braket_sampler)

#     chainstrength = 8
#     numruns = 10

#     response = sampler.sample_qubo(Q, chain_strength=chainstrength, num_reads=numruns)
#     energies = iter(response.data())



class BaseChannel(ABC):
    def __init__(self, params, graph):
        self.params = params
        self.graph = graph
        super().__init__()

    @abstractmethod
    def send(self, x_in):
        pass

    @abstractmethod
    def get_nishimori_temperature(self):
        pass

    @abstractmethod
    def conditional_density(self, y_out, y_in):
        pass

    def encode(self, x_in):
        x_in = np.array(x_in)

        list_edges = list(self.graph.edges())

        y = np.zeros(len(list_edges))
        for i, edge in enumerate(list_edges):
            y[i] = x_in[edge[0]] * x_in[edge[1]]

        return y

    def decode(self, y_out, T=0):
        Q = defaultdict(int)
        list_edges = np.array(list(self.graph.edges))

        # Fill in Q matrix
        for u, v in self.graph.edges:
            i_edge = np.where(list_edges == (u,v))[0][0]
            Q[(u,v)] += 0.5 * np.log(self.conditional_density(y_out[i_edge], 1) / self.conditional_density(y_out[i_edge], -1))
            

        sampler = ExactSolver()
        x_dec = np.array(list(sampler.sample_qubo(Q).first.sample.values()))
        x_dec = 1-2*x_dec

        return x_dec


class BinarySymmetricChannel(BaseChannel):
    def __init__(self, p_error, graph):
        params = {"p_error": p_error}

        super().__init__(params, graph)

    def send(self, y_in):
        errors = np.random.binomial(1, self.params['p_error'], size=y_in.shape)
        print("errors", errors)
        return (y_in*(1-2*errors))

    def get_nishimori_temperature(self):
        p = self.params['p_error']
        return 2 / np.log((1-p)/p)

    def conditional_density(self, y_out, y_in):
        if y_out != y_in:
            return self.params['p_error']
        else:
            return 1-self.params['p_error']


if __name__ == "__main__":
    p_error = 0.2
    graph = dnx.chimera_graph(1)
    channel = BinarySymmetricChannel(p_error, graph)

    # x_in = [1 for _ in range(8)]
    x_in = [-1,1,-1,1,-1,1,-1,1]
    print("x_in", x_in)
    y_in = channel.encode(x_in)
    print("y_in", y_in)
    y_out = channel.send(y_in)
    print("y_out", y_out)
    y_dec = channel.decode(y_out)
    print("y_dec", y_dec)
    

