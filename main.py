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
        self.list_edges = np.array(graph.edges)

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

        n_edges = len(self.list_edges)

        y = np.zeros(n_edges+len(x_in))
        for i, edge in enumerate(self.list_edges):
            y[i] = x_in[edge[0]] * x_in[edge[1]]
        y[n_edges:] = x_in

        return y

    def decode(self, y_out, T=0):
        J = defaultdict(int)
        self.list_edges = np.array(list(self.graph.edges))
        n_edges = len(self.list_edges)

        for i_edge, (u, v) in enumerate(self.graph.edges):
            # Q[(u,v)] -= 0.5 * np.log(self.conditional_density(y_out[i_edge], 1) / self.conditional_density(y_out[i_edge], -1))
            J[(u,v)] -= y_out[i_edge]

        h = -y_out[n_edges:]

        sampler = ExactSolver()
        x_dec = np.array(list(sampler.sample_ising(h, J).first.sample.values()))
        
        return x_dec

    def get_ber(self, x_in, x_dec):
        N = len(x_in)
        return np.sum(np.abs(x_in - x_dec)) / (2*N)


class BinarySymmetricChannel(BaseChannel):
    def __init__(self, p_error, graph):
        params = {"p_error": p_error}

        super().__init__(params, graph)

    def send(self, y_in):
        errors = np.random.binomial(1, self.params['p_error'], size=y_in.shape)
        # print("errors", errors)
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
    n_p = 40
    n_reps = 100
    graph = dnx.chimera_graph(1)
    x_in = np.array(np.ones(len(graph.nodes)))

    ber = np.zeros((n_p, n_reps))
    list_p = np.linspace(0.01, 0.499, n_p)
    for i_p, p_error in enumerate(list_p):
        print("================= p_error", p_error)
        channel = BinarySymmetricChannel(p_error, graph)

        for i_rep in range(n_reps):
            y_in = channel.encode(x_in)
            # print("y_in", y_in)
            y_out = channel.send(y_in)
            # print("y_out", y_out)
            x_dec = channel.decode(y_out)
            # print("x_dec", x_dec)
            ber[i_p, i_rep] = channel.get_ber(x_in, x_dec)
            # print(ber[i_p, i_rep])
    
    plt.xlabel("Crossover probability")
    plt.ylabel("BER (T=0)")
    plt.plot(list_p, np.mean(ber, axis=1))
    plt.show()
