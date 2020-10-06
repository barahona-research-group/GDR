
Graph Diffusion Reclassification
=====================

Graph diffusion reclassification (GDR) is an algorithm for graph semi-supervised learning.

Classification tasks based on feature vectors can be significantly improved by including within deep learning a graph that summarises pairwise relationships between the samples. Intuitively, the graph acts as a conduit to channel and bias the inference of class labels. Here, we study classification methods that consider the graph as the originator of an explicit graph diffusion. We show that appending graph diffusion to feature-based learning as an a posteriori refinement achieves state-of-the-art classification accuracy. This method, which we call Graph Diffusion Reclassification (GDR), uses overshooting events of a diffusive graph dynamics to reclassify individual nodes. The method uses intrinsic measures of node influence, which are distinct for each node, and allows the evaluation of the relationship and importance of features and graph for classification. 

The code in this repository implements the GDR algorithm and contains a small pipeline to run examples on benchmark datasets. 

For more information please see our published [paper](https://www.aimsciences.org/article/doi/10.3934/fods.2020002).


## Cite

Please cite our paper if you use this code in your own work:

```
Semi-supervised classification on graphs using explicit diffusion dynamics,Foundations of Data Science,2,1,19,33,2020-2-11,Robert L. Peach,Alexis Arnaudon,Mauricio Barahona
```

The bibtex code:

```
@article{peach2019semi,
  title={Semi-supervised classification on graphs using explicit diffusion dynamics},
  author={Peach, Robert L and Arnaudon, Alexis and Barahona, Mauricio},
  journal={arXiv preprint arXiv:1909.11117},
  year={2019}
}

```

## Installation

Require packages include:
- python3
- scipy/numpy/matplotlib
- networkx
- sklearn 

To install, run in the main folder:

```python setup.py install```

or 

```pip install .```


## Tests

In the test folder, the script ``run.py`` can be used to run the benchmark examples in the paper.

Simply change the dataset variable to one of:
* ``cora``
* ``wikipedia``
* ``pubmed``
* ``citeseer``
* ``cora-d``  (for directed cora)

Alternatively, you can apply the methodology to your own data.



## Our other available packages

If you are interested in trying our other packages, see the below list:
* [MSC](https://github.com/barahona-research-group/MultiscaleCentrality) : MultiScale Centrality: A scale dependent metric of node centrality.
* [hcga](https://github.com/barahona-research-group/hcga) : Highly comparative graph analysis. A graph analysis toolbox that performs massive feature extraction from a set of graphs, and applies supervised classification methods.





