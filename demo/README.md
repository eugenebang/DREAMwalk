## Input File formats

### Hierarchy file

The input hierarchy file for semantic similarity calculation should be a `.csv` file with following information:
| child | parent  |
| ------- | --- |
| child1 | parent1 |
| parent1 | parent2 |
| ... | ... | 

_Caution:_ All leafs should be branched out from root named as either 'disease' or 'drug'

### Network file

The input network file should be a tab- or space-separated `.txt` file with following orders:
```python
source target weight edge_type edge_id
```

### Positive/negative pairs

Positive/negative pair information for link prediction task should be 
