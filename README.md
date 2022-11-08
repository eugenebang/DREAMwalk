    .
    ├── DREAMwalk/                          # Codes for DREAMwalk
            ├── add_similarity_edges.py     
                ├── read_csv
                ├── jaccard_index
                ├── read_graph
            ├── generate_embedding.py       
                ├── read_graph
                ├── read_edge_type_matrix
                ├── generate_tp_paths
                ├── parmap_DREAMwalks
                ├── node2vecTP_walk
                ├── network_traverse
                ├── teleport_operation
            ├── predict_association.py      
            └── utils.py                    # func for ranking query drugs from demo_repurposing.txt
                ├── set_seed
                ├── read_graph
    ├── demo/                               # demo files
            ├── demo_hierarchy.csv
            ├── demo_network.txt
            ├── demo_edges.csv
            └── demo_repurposing.txt
    ├── example.ipynb                       # Notebook with example codes for DREAMwalk pipeline
    └── README.md

# DREAMwalk
The official code implementation for DREAMwalk in Python from our paper: 
- "Multi-layer Guilt-by-association for drug repurposing"

We also provide codes for calculating semantic similarities of entities given hierarchy data, along with example heterogeneous network file and notebook file for generating node embedding and predicting associations.

## Model description

The full model architecture is provided below. DREAMwalk's drug-disease association prediction pipeline is consisted of three steps;

Step 1. Edge type transition matrix training

Step 2. Node embedding generation through teleport-guided random walk

Step 3. Drug-disease association prediction (or prediction of any other links)

![model1](img/model_overview.png)


## Run the model
1. If similarity network for entities are required, run the following:
```
$ python add_similarity_edges.py \
    --hierarchy_file=hierarchy_file.csv \
    --network_file=input_network.txt \
    --cut_off=0.5
```

2. For embedding vector generation, with input network file for embedding:
```
$ python generate_embeddings.py  \
    --network_file=input_network.txt \
    --sim_network_file=sim_network.txt \
    --output_file=embedding_file.pkl \
    --tp_factor=0.5 \
    --sim_edge_index=[1,2]
```

3. For downstream linkprediction task please run: 
```
$ python predict_association.py \
    --embedding_file=embedding_file.pkl \
    --pair_file=dataset.csv \
    --output_file=predicted_output.csv
```

The file formats for each input file can be found [here](demo/README.md).

### Software requirements

**Operating system**

DREAMwalk training and evaluation were tested for *Linux* (Ubuntu 18.04) operating systems.

**Prerequisites**

DREAMwalk training and evaluation were tested for the following python packages and versions.

- **For embedding generation**
  - `python`=3.9.6
  - `gensim`=1.????
  - `networkx`=1.???
  - `numpy`=1.21.0
  - `pandas`=1.3.0
  - `scikit-learn`=0.24.2
  - `tqdm`=4.61.2
  - `parmap`=1.???
- **Additionally requried for link prediction**
  - `pytorch`=1.9.0 (installed using cuda version 11.1)
