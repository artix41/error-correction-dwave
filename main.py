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
from tqdm import tqdm
from scipy.stats import binom
import os


def take_subgraph(sampler, n_nodes):
    list_nodes = np.array(sampler.nodelist)

    new_nodes = list_nodes.copy()
    type_line = 0
    while len(new_nodes) > n_nodes:
        layout = dnx.pegasus_layout(sampler.to_networkx_graph().subgraph(new_nodes))

        values = np.vstack(layout.values())
        min_x = np.min(values[:,0])
        max_x = np.max(values[:,0])
        min_y = np.min(values[:,1])
        max_y = np.max(values[:,1])

        if type_line == 0:                
            new_new_nodes = np.array([item[0] for item in layout.items() if item[1][0] > min_x])
        elif type_line == 1:                
            new_new_nodes = np.array([item[0] for item in layout.items() if item[1][0] < max_x])
        elif type_line == 2:                
            new_new_nodes = np.array([item[0] for item in layout.items() if item[1][1] > min_y])
        elif type_line == 3:                
            new_new_nodes = np.array([item[0] for item in layout.items() if item[1][1] < max_y])

        if len(new_new_nodes) < n_nodes:
            if type_line == 0:                
                nodes_to_delete = [item[0] for item in layout.items() if item[1][0] == min_x]
            elif type_line == 1:                
                nodes_to_delete = [item[0] for item in layout.items() if item[1][0] == max_x]
            elif type_line == 2:                
                nodes_to_delete = [item[0] for item in layout.items() if item[1][1] == min_y]
            elif type_line == 3:                
                nodes_to_delete = [item[0] for item in layout.items() if item[1][1] == max_y]
    
            new_nodes = new_nodes[~np.isin(new_nodes, nodes_to_delete[:len(new_nodes) - n_nodes])]
        else:
            new_nodes = new_new_nodes

        type_line = (type_line + 1) % 4

    return new_nodes

class BaseChannel(ABC):
    def __init__(self, params, graph_type="chimera", n_nodes=None, n_edges=None):
        self.params = params
        self.graph_type = graph_type
        device_name = {'chimera': 'DW_2000Q_6', 'pegasus': 'Advantage_system1.1', 'simulator': None}

        if graph_type == 'simulator':
            self.sampler = DWaveSampler(solver={'qpu': False})
        else:
            self.sampler = DWaveSampler(solver={'qpu': True, 'name': device_name[graph_type]})

        self.list_edges = np.array(self.sampler.edgelist)
        self.list_nodes = np.array(self.sampler.nodelist)
        
        nodes_file = "data/pegasus/list_nodes.npy"
        edges_file = "data/pegasus/list_edges.npy"
        if graph_type == 'pegasus':
            if n_nodes is not None:
                if os.path.exists(nodes_file):
                    print("Loading list nodes...")
                    self.list_nodes = np.load(nodes_file)
                else:
                    print("Taking subgraph...")
                    self.list_nodes = take_subgraph(self.sampler, n_nodes)
                    np.save(nodes_file, self.list_nodes)

                self.graph = self.sampler.to_networkx_graph().subgraph(self.list_nodes)
                self.list_edges = np.array(self.graph.edges)
            if n_edges is not None:
                if os.path.exists(edges_file):
                    print("Loading list edges...")
                    self.list_edges = np.load(edges_file)
                else:
                    print("Removing edges...")
                    edges_idx = np.sort(np.random.choice(len(self.list_edges), n_edges, replace=False))
                    self.list_edges = self.list_edges[edges_idx]
                    np.save(edges_file, self.list_edges)

        print("Number of qubits", len(self.list_nodes))
        print("Number of edges", len(self.list_edges))
        
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

    def decode(self, J_out, h_out, num_reads=1, num_spin_reversals=1):
        result = self.sampler.sample_ising(h_out, J_out, num_reads=num_reads, num_spin_reversal_transforms=num_spin_reversals)
        x_dec = np.array(list(list(result.data())[0].sample.values()))

        return x_dec

    def get_ber(self, x_in, x_dec):
        N = len(x_in)
        return np.sum(np.abs(x_in - x_dec)) / (2*N)


class BinarySymmetricChannel(BaseChannel):
    def __init__(self, p_error, graph_type, n_nodes=None, n_edges=None):
        params = {"p_error": p_error}

        super().__init__(params, graph_type=graph_type, n_nodes=n_nodes, n_edges=n_edges)

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


def plot_ber(ber_file):
    ber = np.load(ber_file)
    n_p, n_reps = ber.shape

    list_p = np.linspace(0.01, 0.499, n_p)
    plt.xlabel("Crossover probability")
    plt.ylabel("BER (T=0)")
    plt.plot(list_p, np.mean(ber, axis=1))

    plt.savefig("images/ber.png", bbox_inches='tight')

if __name__ == "__main__":
    n_p = 10
    n_reps = 1
    graph_type = 'pegasus'
    num_spin_reversals = 0
    num_reads = 100
    n_nodes = 2041
    n_edges = 5974

    channel = BinarySymmetricChannel(0.001, graph_type, n_nodes=n_nodes, n_edges=n_edges)

    x_in = np.array(np.ones(len(channel.list_nodes)))
    ber = np.zeros((n_p, n_reps))
    list_p = np.linspace(0.01, 0.499, n_p)

    print(f"Graph type: {graph_type}")
    for i_p, p_error in enumerate(list_p):
        print(f"================= p: {p_error} ==================")
        channel = BinarySymmetricChannel(p_error, graph_type, n_nodes=n_nodes, n_edges=n_edges)

        for i_rep in range(n_reps):
            J_in, h_in = channel.encode(x_in)
            J_out, h_out = channel.send(J_in, h_in)
            x_dec = channel.decode(J_out, h_out, num_reads=num_reads, num_spin_reversals=num_spin_reversals)
            ber[i_p, i_rep] = channel.get_ber(x_in, list(x_dec))
            print(ber[i_p, i_rep])

    ber_file = f"data/{graph_type}/ber-{num_reads}-reads-{num_spin_reversals}-sr-{len(channel.list_edges)}-edges.npy"

    np.save(ber_file, ber)

    plot_ber(ber_file)
    # plt.show()
