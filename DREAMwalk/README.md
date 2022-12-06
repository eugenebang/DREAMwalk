## Run the model
1. To generate similarity network for drug/disease entities using hierarchy data, run the following:
```
$ python generate_similarity_net.py \
    --network_file input_network.txt \
    --hierarchy_file hierarchy_file.csv \
    --output_file sim_network.txt \
    --cut_off 0.5
```

2. For embedding vector generation, with input network file for embedding:
```
$ python generate_embeddings.py\
    --network_file input_network.txt \
    --sim_network_file input_similarity_network.txt \
    --dimension 5
```

3. For downstream linkprediction task please run: 
```
$ python predict_associations.py \
    --embedding_file embedding_file.pkl \
    --pair_file dataset.csv
```

The file formats for each input file can be found [here](/demo/README.md).

