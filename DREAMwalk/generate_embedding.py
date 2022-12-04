import argparse

import os
import math
import random
import pickle
import networkx as nx
import numpy as np   
from scipy import stats
from tqdm import tqdm
import parmap

from gensim.models import Word2Vec

from utils import read_graph

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_file', type=str, required=True)
    
    parser.add_argument('--sim_network_file', type=str, default=False)
    parser.add_argument('--output_file', type=str, default='embedding_file.pkl')
    parser.add_argument('--tp_factor', type=float, default=0.5)
    parser.add_argument('--weighted', type=bool, default=True)    
    parser.add_argument('--directed', type=bool, default=False)
    parser.add_argument('--num_walks', type=int, default=100)
    parser.add_argument('--walk_length', type=int, default=10)
    parser.add_argument('--dimension', type=int, default=128)    
    parser.add_argument('--window_size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=os.cpu_count(),
                       help='if default, set to all available cpu count')
    parser.add_argument('--net_delimiter',type=str,default='\t',
                       help='delimiter of networks file; default = tab')
    return parser.parse_args()

def generate_DREAMwalk_paths(G, G_sim, num_walks, walk_length,tp_factor, workers):
    '''
    generate random walk paths constrainted by transition matrix
    '''
    tot_walks = []
    nodes = list(G.nodes())

    print(f'# of walkers : {min(num_walks,workers)} for {num_walks} times')
    walks = parmap.map(_parmap_walks, range(num_walks), 
                        nodes, G, G_sim, walk_length,tp_factor,
                        pm_pbar=True, pm_processes=min(num_walks,workers))
    for walk in walks:
        tot_walks += walk

    return tot_walks

def _parmap_walks(_, nodes, G, G_sim, walk_length,tp_factor):
    walks = []
    random.shuffle(nodes) 
    for node in nodes:
        walks.append(_DREAMwalker(node, G, G_sim, walk_length,tp_factor))
    return walks

def _network_traverse(cur,G):
    cur_nbrs = sorted(G.neighbors(cur))
    random.shuffle(cur_nbrs)
    selected_nbrs=[]
    distance_sum=0
    for nbr in cur_nbrs:
        nbr_links = G[cur][nbr]
        for i in nbr_links:
            nbr_link_weight = nbr_links[i]['weight']
            distance_sum += nbr_link_weight
            selected_nbrs.append(nbr)
    rand = np.random.rand() * distance_sum
    threshold = 0
    
    for nbr in set(selected_nbrs):
        nbr_links = G[cur][nbr]
        for i in nbr_links:
            nbr_link_weight = nbr_links[i]['weight']
            threshold += nbr_link_weight
            if threshold >= rand:
                next = nbr
                break
    return next    

def _teleport_operation(cur,G):
    cur_nbrs = sorted(G.neighbors(cur))
    random.shuffle(cur_nbrs)
    selected_nbrs=[]
    distance_sum=0
    for nbr in cur_nbrs:
        nbr_links = G[cur][nbr]
        for i in nbr_links:
            nbr_link_weight = nbr_links[i]['weight']
            distance_sum += nbr_link_weight
            selected_nbrs.append(nbr)
    if distance_sum==0:
        return False
    rand = np.random.rand() * distance_sum
    threshold = 0

    for nbr in set(selected_nbrs):
        nbr_links = G[cur][nbr]
        for i in nbr_links:
            nbr_link_weight = nbr_links[i]['weight']
            threshold += nbr_link_weight
            if threshold >= rand:
                next = nbr
                break
    return next

def _DREAMwalker(start_node, G, G_sim, walk_length,tp_factor):
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        if np.random.rand() < tp_factor:
            next_node=_teleport_operation(cur,G_sim)
            if not next_node:  # node not in G_sim
                next_node=_network_traverse(cur,G)                
        else:
            next_node=_network_traverse(cur,G)
        walk.append(next_node)
    return walk

if __name__ == '__main__':
    args=parse_args()
    print('Reading network files...')
    G=read_graph(args.network_file,
                weighted=args.weighted,
                directed=args.directed)
    if args.sim_network_file:
        G_sim=read_graph(args.sim_network_file,
                        weighted=True,
                        directed=False)
        tp_factor=args.tp_factor
    else:
        G_sim=None
        print('No input similarity file detected')
        tp_factor=0
    walks=generate_DREAMwalk_paths(G,G_sim,args.num_walks,args.walk_length,
                                  tp_factor, args.workers)
    print('END of path generation')
    
    print('Generating node embeddings...')
    model = Word2Vec(walks, vector_size = args.dimension, window= args.window_size, 
                     min_count=0, sg=1, workers= args.workers)
    node_ids=model.wv.index_to_key
    node_embeddings=model.wv.vectors

    model_embeddings={}
    for i,idx in enumerate(node_ids):
        model_embeddings[idx]=node_embeddings[i]
    with open(output_file,'wb') as fw:
        pickle.dump(model_embeddings,fw)
    print(f'Node embeddings saved in: {args.output_file}')
