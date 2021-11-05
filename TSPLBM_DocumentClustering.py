# -*- coding: utf-8 -*-

"""
This code allows us to run TSPLBM algorithm on all datasets
"""
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
import sparseTensorCoclustering
from coclust.evaluation.external import accuracy
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics
import time
import random

def random_init(n_clusters, n_cols, random_state=None):
    """Create a random column cluster assignment matrix.
    Each row contains 1 in the column corresponding to the cluster where the
    processed data matrix column belongs, 0 elsewhere.
    Parameters
    ----------
    n_clusters: int
        Number of clusters
    n_cols: int
        Number of columns of the data matrix (i.e. number of rows of the
        matrix returned by this function)
    random_state : int or :class:`numpy.RandomState`, optional
        The generator used to initialize the cluster labels. Defaults to the
        global numpy random number generator.
    Returns
    -------
    matrix
        Matrix of shape (``n_cols``, ``n_clusters``)
    """

    if random_state == None:
        W_a = np.random.randint(n_clusters, size=n_cols)

    else:
        random_state = check_random_state(random_state)
        W_a = random_state.randint(n_clusters, size=n_cols)

    W = np.zeros((n_cols, n_clusters))
    W[np.arange(n_cols), W_a] = 1
    return W


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


global_path = './data/'
path_to_save = './results/ResultTSPLBM/'

nom_BDD = ['DBLP1','DBLP2','PubMed_Diabets' ,'classic3', 'classic4', 'ag_news']

nCLusters = [3,3,3,3, 4,  4]


nbrIteration = 30
config = [['BOW', 'Bert', 'SentenceRo', 'GLOVE', 'Entity']]
for c in range(len(config)):

    slices = config[c]
    subname = ''
    for s in range(len(slices)):
        subname = subname + slices[s] + '_'

    df_results = pd.DataFrame(columns=["Dataset", "Time", "ACC", "NMI", "ARI", "Purity"],
                              index=np.arange(nbrIteration * ((len(nom_BDD)))).tolist())
    cpt = 0
    for nb in range(len(nom_BDD)):
        print('###############################################')
        print('nom_BDD ', nom_BDD[nb])
        bdd_name = nom_BDD[nb]
        inf_doc = pd.read_csv(global_path+bdd_name+'/' + bdd_name+'.csv', delimiter=',')
        abstracts = np.asarray(inf_doc['text']).astype(str).tolist()
        # print(inf_doc)
        labels_all = np.asarray(inf_doc['label']).astype(int)

        n_new = inf_doc.shape[0]
        d_new = inf_doc.shape[1]

        labels = labels_all[0:n_new]
        labels = labels.tolist()

        ##################################################################
        #                            hyperparameters                     #
        ##################################################################

        K = nCLusters[nb]
        print('K ', K)
        del inf_doc

        ##################################################################
        #                        Load DBLP1 dataset                      #
        ##################################################################
        simBow = np.load(global_path +bdd_name+'/'  + 'sim_bow' + '.npz')
        simBow = simBow['arr_0']

        simBow2 = np.load(global_path +bdd_name+'/'  + 'sim_bow_v2' + '.npz')
        simBow2 = simBow2['arr_0']

        simBert = np.load(global_path +bdd_name+'/'  + 'sim_bert-base-cased' + '.npz')
        simBert = simBert['arr_0']

        simBertLPCA = np.load(global_path + bdd_name + '/' + 'sim_avgpca__bert-large-cased' + '.npz')
        simBertLPCA = simBertLPCA['arr_0']

        simRoBertLPCA = np.load(global_path + bdd_name + '/' + 'sim_avgpca__roberta-large' + '.npz')
        simRoBertLPCA = simRoBertLPCA['arr_0']

        simRoBerta = np.load(global_path +bdd_name+'/'  + 'sim_roberta-large' + '.npz')
        simRoBerta = simRoBerta['arr_0']

        simSentenceRoBerta = np.load(global_path +bdd_name+'/'  + 'sim_sentenceRoberta' + '.npz')
        simSentenceRoBerta = simSentenceRoBerta['arr_0']

        simGlove = np.load(global_path +bdd_name+'/'  + 'sim_glove' + '.npz')
        simGlove = simGlove['arr_0']

        simW2V = np.load(global_path +bdd_name+'/'  + 'sim_w2v' + '.npz')
        simW2V = simW2V['arr_0']

        simEntity = np.load(global_path +bdd_name+'/'  + 'sim_entity' + '.npz')
        simEntity = simEntity['arr_0']

        data = []
        for s in range(len(slices)):
            if slices[s] == 'BOW':
                data.append(simBow)

            if slices[s] == 'BOW2':
                data.append(simBow2)

            if slices[s] == 'Bert':
                data.append(simBertLPCA)

            if slices[s] == 'BertLPCA':
                data.append(simBertLPCA)

            if slices[s] == 'RoBertaLPCA':
                data.append(simRoBertLPCA)

            if slices[s] == 'RoBerta':
                data.append(simRoBerta)

            if slices[s] == 'GLOVE':
                data.append(simGlove)

            if slices[s] == 'Entity':
                data.append(simEntity)

            if slices[s] == 'SentenceRo':
                data.append(simSentenceRoBerta)

        print(len(data))
        del simBow
        del simGlove
        del simW2V
        del simBert
        del simEntity
        del simBertLPCA
        del simRoBertLPCA
        ##################################################################
        ########################## Version Hard #########################
        ##################################################################
        data = np.dstack(data)
        print('data ', data.shape)
        for it in range(nbrIteration):
            random.seed(it)
            np.random.seed(it)
            print("iter " + str(it))
            Z_init = random_init(K, n_new)

            start_time = time.time()
            model = sparseTensorCoclustering.SparseTensorCoclusteringPoisson(n_clusters=K,  fuzzy = False, init_row=Z_init, init_col=Z_init, max_iter=50)
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

            df_results.Dataset[cpt] = bdd_name
            df_results.Time[cpt] = str(time_)
            df_results.ACC[cpt] = str(acc)
            df_results.NMI[cpt] = str(nmi)
            df_results.ARI[cpt] = str(ari)
            df_results.Purity[cpt] = str(purity)
            cpt = cpt + 1


    df_results.to_csv(path_to_save + "Results_TSPLBM_" + subname + ".csv", index=False)
