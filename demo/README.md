## Example codes
Sample code to generate the embedding space and predict drug-disease associations are provided in [`run_demo.ipynb`](../run_demo.ipynb).

## Input File formats

### Hierarchy file

The input hierarchy file for semantic similarity calculation should be a `.csv` file with following information:
| child | parent  |
| ------- | --- |
| child1 | parent1 |
| parent1 | parent2 |
| parent2 | disease |
| ... | ... | 
| parent n | drug |
| ... | ... | 

_Caution:_ All leafs should be branched out from root named as either 'disease' or 'drug'

### Network file

The input network file should be a tab- or space-separated `.txt` file with following orders:
```python
source target weight edge_type edge_id
```

### Node type label file

The input node type label file for heterogeneous Skip-gram should be a '.tsv' file with following information:
| node | type  |
| ------- | --- |
| node 1 | gene |
| node 2 | disease |
| node 3 | gene |
| ... | ... | 
| node n | drug |
| ... | ... | 

_Caution:_ The drug, disease and gene node types should be given as `drug`, `disease`, `gene` for the model to properly recognize its type.

### Positive/negative drug-disease association file

Positive/negative pair information for drug-disease association prediction task should be input as a `.tsv` file with following structure:
| drug | disease| label |
| --- | --- | --- |
| drug1 | disease1 | 1 |
| drug2 | disease2 | 0 | 
| ... | ... | ... |

where `label == 1` refers to positive pairs and `label == 0` refers to negative pairs.
