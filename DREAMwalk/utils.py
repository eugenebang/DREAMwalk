import os
import random
import numpy as np
import networkx as nx


def read_graph(edgeList,weighted=True, directed=False,delimiter='\t'):
    '''
    Reads the input network in networkx.
    '''
    if weighted:
        G = nx.read_edgelist(edgeList, nodetype=str, 
                             data=(('type',int),('weight',float),('id',int)), 
                             create_using=nx.MultiDiGraph(),
                            delimiter=delimiter)
    else:
        G = nx.read_edgelist(edgeList, nodetype=str,data=(('type',int)), 
                             create_using=nx.MultiDiGraph(),
                            delimiter=delimiter)
        for edge in G.edges():
            edge=G[edge[0]][edge[1]]
            for i in range(len(edge)):
                edge[i]['weight'] = 1.0

    if not directed:
        G = G.to_undirected()

    return G

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f'random seed with {seed}')