import os
import torch
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'random seed with {seed}')
    
class EarlyStopper:
    """Stop training when validation loss shows no improvement throughout patience epochs"""
    def __init__(self, patience=7, printfunc=print,verbose=False, delta=0, path='checkpoint.pt'):
        self.printfunc=printfunc
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.printfunc(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''saves model when validation loss is decreased'''
        if self.verbose:
            self.printfunc(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
