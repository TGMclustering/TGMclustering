# TGM 

<p align="justify">
TGM (Tensor-based Graph Modularity) is a novel algorithm aiming to cluster tensor data.
It is a general, practical, and parameter-free method to learn a consensus clustering from multiple graphs via tensor-based graph modularity. The advantage of the proposed approach is taking 
into account the graphs' properties (degree of nodes, edges density, community structure, etc.)  through the modularity measure. 
We propose to apply the proposed approach to text clustering. A suitable way of combining the different available representations consists in using them as part of a 3-way similarity tensor.
To achieve that for a given dataset, we first compute different representation matrices (using BOW,  static embeddings such GloVe, contextual embeddings such BERT, and entity linking). Then, we compute from each data representation a similarity matrix viewed as the adjacency matrix of a graph that connects the documents (nodes). 
Finally, the similarity matrices are structured as a 3-way tensor (cf. figure below)

It also allows to easily perform tensor clustering through decomposition or tensor learning and tensor algebra. 
TGMclustering allows easy interaction with other python packages such as NumPy, Tensorly, TensorFlow, or TensorD, and run methods at scale on CPU or GPU.

**It supports major operating systems namely Microsoft Windows, macOS, and Ubuntu**.
</p>

[![N|Solid](https://github.com/TGMclustering/TGMclustering/blob/main/diagram-TGM.png?raw=true)]()

- Source-code: https://github.com/TGMclustering/TGMclustering

- GPU version is available at https://github.com/TGMclustering/TGMclustering/blob/main/TensorClustModSparseGPU.py

### Requirements

All needed packages are available in the [requirements file](https://github.com/TGMclustering/TGMclustering/blob/main/requirements.txt)


### Using TGM

To clone TGMclustering project from github
```
# Install git LFS via https://www.atlassian.com/git/tutorials/git-lfs
# initialize Git LFS
git lfs install Git LFS initialized.
git init Initialized
# clone the repository
git clone https://github.com/TGMclustering/TGMclustering.git
cd TGMclustering
```

### License
TGM is released under the MIT License (refer to LISENSE file for details).

### Examples

```python
import sparseTensorCoclustering as tcSCoP
import numpy as np
import pandas as pd
import TGMclustering
from coclust.evaluation.external import accuracy
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics
import time


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


global_path = './data/'
path_to_save = './results/ResultTGM/'
path_to_save_mod = './results/ResultTGM/Modularity/'
path_to_save_partition = './results/ResultTGM/Partition/'

nom_BDD = 'DBLP1'

nbrIteration = 30
config = ['BOW', 'Bert', 'SentenceRo', 'GLOVE', 'Entity']
subname = ''
for s in range(len(config)):
    subname = subname + config[s] + '_'

bdd_name = nom_BDD
inf_doc = pd.read_csv(global_path + bdd_name + '/' + bdd_name + '.csv', delimiter=',')
abstracts = np.asarray(inf_doc['text']).astype(str).tolist()
labels_all = np.asarray(inf_doc['label']).astype(int)
labels = labels_all.tolist()

##################################################################
#                            hyperparameters                     #
##################################################################

K = 3
print('K ', K)
del inf_doc

##################################################################
#                        Load DBLP1 dataset                      #
##################################################################
simBow = np.load(global_path + bdd_name + '/' + 'sim_bow' + '.npz')
simBow = simBow['arr_0']

simBertLPCA = np.load(global_path + bdd_name + '/' + 'sim_avgpca__bert-large-cased' + '.npz')
simBertLPCA = simBertLPCA['arr_0']

simSentenceRoBerta = np.load(global_path + bdd_name + '/' + 'sim_sentenceRoberta' + '.npz')
simSentenceRoBerta = simSentenceRoBerta['arr_0']

simGlove = np.load(global_path + bdd_name + '/' + 'sim_glove' + '.npz')
simGlove = simGlove['arr_0']

simEntity = np.load(global_path + bdd_name + '/' + 'sim_entity' + '.npz')
simEntity = simEntity['arr_0']

data = [simBow, simBertLPCA, simSentenceRoBerta, simGlove, simEntity]

del simBow
del simGlove
del simEntity
del simBertLPCA
del simSentenceRoBerta
##################################################################
########################## TGM clustering ########################
##################################################################

for it in range(nbrIteration):
    print("Run " + str(it))

    start_time = time.time()
    model = TGMclustering.TGM(n_clusters=K, bool_fuzzy=False,
                                           fusion_method='Add')
    model.fit(data)
    end_time = time.time()

    phiK = model.row_labels_
    phiK = np.asarray(phiK)

    time_ = end_time - start_time
    acc = np.around(accuracy(labels, phiK), 3)
    nmi = np.around(normalized_mutual_info_score(labels, phiK), 3)
    ari = np.around(adjusted_rand_score(labels, phiK), 3)
    purity = np.around(purity_score(labels, phiK), 3)
    print("Accuracy : ", acc)
    print("nmi : ", nmi)
    print("ari : ", ari)
    print("purity : ", purity)
    rt
    accuracy
```
### To reproduce the paper results 
```
Step 1: Clone the GitHub repository of TGMclustering.
Step 2: Install requirements packages.
Step 3: Run TGM_experiments.py to generate the results of TGM for all configurations.
Step 4: Run ConfigurationAnalysis.py to analyze and compare the configurations.
Step 5: Run ViewDocumentClustering.py, SliceDocumentClustering.py,
TensorDecompDocumentClustering.py, and TSPLBM_DocumentClustering.py to generate the results
of the classical, graph, tensor decomposition, and TSPLBM clustering approaches, respectively. 
Step 6: Run TGMLatexTable.py to generate the latex table with all results.
Step 7: Run objectiveFunctionAnalysis to analyze the evolution of the TGM objective
function according to NMI.
Step 8: Run tgm_ensemble_clustering to compare the results of TGM with and without consensus.
Step 9: Run ClusterNumberSelection.py and ClusterNumberSelectionDraw.py to generate
the results of the cluster number selection using the TGM objective function. 
Step 10: Run TimeAnalysis_TGM_vs_TSPLBM to compare TGM and TSPLBM in terms of time execution.
```