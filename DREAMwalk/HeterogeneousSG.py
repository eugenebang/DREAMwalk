import os
import numpy as np
import pickle

def _prep_hetSG_walks(walks, node2id, nodetypef):
    with open(nodetypef,'r') as fr:
        lines=fr.readlines()
    node2type={}
    for line in lines[1:]:
        line=line.strip().split('\t')
        node2type[line[0]]=line[1]

    # we need to annotate the nodes with prefixes for the HetSG to recognize the nodetype
    type2meta = {
        'drug':'d',
        'disease':'i',
        'gene':'g',
        'etc':'e'
            }

    annot_walks=[]
    for walk in walks:
        annot_walk=[]
        for node in walk:
            nodeid=node2id[node]
            try : node = type2meta[node2type[node]]+nodeid
            except KeyError : node='e'+nodeid
            annot_walk.append(node)
        annot_walks.append(' '.join(annot_walk))
    return annot_walks

def _prep_SG_walks(walks, node2id):
    # does not require node type prefixes
    annot_walks=[]
    for walk in walks:
        annot_walk=[node2id[node] for node in walk]
        annot_walks.append(' '.join(annot_walk))        
    return annot_walks

def HeterogeneousSG(use_hetSG:bool, walks:list, nodes:set, nodetypef:str, embedding_size:int, 
                   window_length:int, workers:int):
    use_hetSG=int(use_hetSG)
    node2id={node:str(i) for i, node in enumerate(nodes)}
    id2node={str(i):node for i, node in enumerate(nodes)}
    
    if use_hetSG:   
        annot_walks = _prep_hetSG_walks(walks,node2id, nodetypef)
    else:
        annot_walks = _prep_SG_walks(walks,node2id)
        
    tmp_walkf='tmp_walkfile'
    with open(tmp_walkf,'w') as fw:
        fw.write('\n'.join(annot_walks))

    outputf = 'tmp_outputf'
        
    # use hetsg.cpp file for embedding vector generation -> outputs .txt file
    os.system('g++ DREAMwalk/HeterogeneousSG.cpp -o HetSG -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result')
    os.system(f'./HetSG -train {tmp_walkf} -output {outputf} -pp {use_hetSG} -min-count 0 -size {embedding_size} -iter 1 -window {window_length} -threads {workers}')
    
    with open(outputf+'.txt','r') as fr:
        lines=fr.readlines()
        
    # remove tmp files
    os.system(f'rm -f {outputf} {tmp_walkf} {outputf}.txt HetSG')
    
    embeddings={}
    for line in lines[1:]:
        line=line.strip().split(' ')
        if line[0] == '</s>':
            pass
        else:
            nodeid = line[0]
            if use_hetSG:
                embedding=np.array([float(i) for i in line[1:]], dtype=np.float32)
                embeddings[id2node[nodeid[1:]]]=embedding
            else:
                embedding=np.array([float(i) for i in line[1:]], dtype=np.float32)
                embeddings[id2node[nodeid]]=embedding

    return embeddings