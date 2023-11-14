# MetaExplainer

### Setup

1. Download [DRKG](https://github.com/gnn4dr/DRKG) or use the following command:

```
wget https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz
```

2. When you untar ```drkg.tar.gz```, you will find ```drkg.tsv```.

3. Add ```drkg.tsv``` to ```Input/DRKG```.

### Quick Tour

* TrainGNN.ipynb : Takes as input ```drkg.tsv``` and outputs a trained Graph Neural Network (can be either GCN, GraphSAGE or GAT) to ```Output/GNNModels```.
* ExplainabilityUseCase.ipynb : Takes as input a model and its precitions, and returns a dictionary (for a specific use case) containing the most important nodes that were most influential for the prediction(s) using Saliency Maps.
* PlotSubgraph.ipynb : Plots the explanatory subgraph.