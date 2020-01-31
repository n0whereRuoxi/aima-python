import networkx as nx
import numpy as np
from objects import *

class Problem():
    def __init__(self, env, n=50, type='simple'):
        self.env = env

        self.graph = None
        if type =='':
            self.graph = nx.empty_graph(n=n)
        elif type == 'simple':
            self.graph = nx.path_graph(n=n)
        elif type == 'acyclic':
            conn_mat = np.random.random((n,n))>.5
            for i in range(n):
                for j in range(n):
                    if i>=j: conn_mat[i,j]=0
            self.graph = nx.convert_matrix.from_numpy_matrix(conn_mat, create_using=nx.DiGraph)


    def dirts_remaining(self):
        s = set(self.graph)

        for a in self.env.agents:
            s -= set(a.holding)

        return list(s)


    def solved(self):
        return bool(self.dirts_remaining())




if __name__ == '__main__':
    import matplotlib.pyplot as plt

    prob = Problem(None,10,type='acyclic')

    print(set(prob.graph))
    nx.draw_networkx(prob.graph,arrows=True)


    plt.show()
