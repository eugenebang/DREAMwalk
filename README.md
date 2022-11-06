# DREAMwalk
The official code implementation for DREAMwalk in Python.

We also provide codes for calculating semantic similarities of entities given hierarchy data, along with example heterogeneous network file and notebook file for generating node embedding and predicting associations.

## Model description

The full model architecture is provided below. DREAMwalk's drug-disease association prediction pipeline is consisted of three steps;
Step 1. Edge type transition matrix training
Step 2. Node embedding generation through teleport-guided random walk
Step 3. Drug-disease association prediction (or prediction of any other links)

![model1](img/model_overview.png)

## Run the model
1. For embedding vector generation, with input network file for embedding:
```
python generate_embeddings.py -i input_network.txt -o embedding_file.pkl
```

2. For downstream linkprediction task, 
```
python predict_association.py -e embedding_file.pkl -d dataset.csv -o predicted_output.csv
```

### File formats

**Hierarchy file**
The input hierarchy file for semantic similarity calculation should be a `.csv` file with following information:
```python
child, parent
```

**Network file**
The input network file should be a tab- or space-separated `.txt` file with following orders:
```python
source target weight edge_type edge_id
```



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
  - 
- **Additionally requried for link prediction**
  - `pytorch`=1.9.0 (installed using cuda version 11.1)
