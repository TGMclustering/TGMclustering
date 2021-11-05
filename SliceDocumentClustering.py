# -*- coding: utf-8 -*-

"""
This code allows us to run graph clustering approaches namely ColcustMod, CoclustInfo and SPLBM
"""
from coclust.coclustering import CoclustInfo
from coclust.coclustering import CoclustMod
import Function_PLBcem_and_SPLBcem as F_PLBvem
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
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
path_to_save = './results/resultSliceClustering/'
nom_BDD = ['DBLP1','DBLP2','PubMed_Diabets' ,'classic3', 'classic4', 'ag_news']

nCLusters = [3,3,3,3, 4,  4]

nbrIteration = 30

nbSlices = 8
nbAlgorithms = 3
df_results = pd.DataFrame(columns=["Dataset", "Time", "ACC", "NMI", "ARI", "Purity",'Algorithm','Slice'],
                          index=np.arange(nbrIteration * ((len(nom_BDD) * nbSlices* nbAlgorithms))).tolist())
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

    simBert = np.load(global_path +bdd_name+'/'  + 'sim_bert-base-cased' + '.npz')
    simBert = simBert['arr_0']

    simBertLPCA = np.load(global_path +bdd_name+'/' + 'sim_avgpca__bert-large-cased' + '.npz')
    simBertLPCA = simBertLPCA['arr_0']

    simRoBertLPCA = np.load(global_path +bdd_name+'/' + 'sim_avgpca__roberta-large' + '.npz')
    simRoBertLPCA = simRoBertLPCA['arr_0']

    simRoBerta = np.load(global_path +bdd_name+'/'  + 'sim_roberta-large' + '.npz')
    simRoBerta = simRoBerta['arr_0']

    simSentenceRoBerta = np.load(global_path +bdd_name+'/' + 'sim_sentenceRoberta' +'.npz')
    simSentenceRoBerta = simSentenceRoBerta['arr_0']

    simGlove = np.load(global_path +bdd_name+'/'  + 'sim_glove' + '.npz')
    simGlove = simGlove['arr_0']

    simW2V = np.load(global_path +bdd_name+'/'  + 'sim_w2v' + '.npz')
    simW2V = simW2V['arr_0']

    simEntity = np.load(global_path +bdd_name+'/'  + 'sim_entity' + '.npz')
    simEntity = simEntity['arr_0']

    viewsNames = ['BOW','Bert','BertLPCA','RoBertLPCA','SentenceRo', 'GLOVE', 'W2V','Entity']
    data = [simBow,simBert,simBertLPCA,simRoBertLPCA,simSentenceRoBerta,simGlove,simW2V,simEntity]
    print(len(data))
    del simBow
    del simGlove
    del simW2V
    del simBert
    del simRoBerta
    del simEntity
    del simBertLPCA
    del simRoBertLPCA
    ##################################################################
    ########################## Version Hard #########################
    ##################################################################
    for v_ in range(len(data)):
        viewName = viewsNames[v_]
        data_view = data[v_]
        print('###############################################')
        print('viewName ',viewName)
        for it in range(nbrIteration):
            print("iter " + str(it))
            random.seed(it)
            np.random.seed(it)
            ##########################################################
            print('################### CoclustInfo ######################')
            Z_init = random_init(K, n_new)

            start_time = time.time()
            model = CoclustInfo(n_row_clusters=K, n_col_clusters=K, init=Z_init,max_iter=100)
            model.fit(data_view)
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

            df_results.Algorithm[cpt] = 'CoclustInfo'
            df_results.Slice[cpt] = viewName
            df_results.Dataset[cpt] = bdd_name
            df_results.Time[cpt] = str(time_)
            df_results.ACC[cpt] = str(acc)
            df_results.NMI[cpt] = str(nmi)
            df_results.ARI[cpt] = str(ari)
            df_results.Purity[cpt] = str(purity)
            cpt = cpt + 1


            ##########################################################
            '''
            print('################### CoclustSpecMod ######################')


            start_time = time.time()
            model = CoclustSpecMod(n_clusters=K,max_iter=100)
            model.fit(data_view)
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

            df_results.Algorithm[cpt] = 'CoclustSpecMod'
            df_results.Slice[cpt] = viewName
            df_results.Dataset[cpt] = bdd_name
            df_results.Time[cpt] = str(time_)
            df_results.ACC[cpt] = str(acc)
            df_results.NMI[cpt] = str(nmi)
            df_results.ARI[cpt] = str(ari)
            df_results.Purity[cpt] = str(purity)
            cpt = cpt + 1
            '''
            ##########################################################
            print('################### CoclustMod ######################')

            start_time = time.time()
            model = CoclustMod(n_clusters=K,init=Z_init,max_iter=100)
            model.fit(data_view)
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

            df_results.Algorithm[cpt] = 'CoclustMod'
            df_results.Slice[cpt] = viewName
            df_results.Dataset[cpt] = bdd_name
            df_results.Time[cpt] = str(time_)
            df_results.ACC[cpt] = str(acc)
            df_results.NMI[cpt] = str(nmi)
            df_results.ARI[cpt] = str(ari)
            df_results.Purity[cpt] = str(purity)
            cpt = cpt + 1
            ##########################################################
            print('################### SPLBM ######################')

            start_time = time.time()
            [pi_k_hat, rho_l_hat, mukl_hat, part, part2, news, acc_ex, nmi_ex, ari_ex] = F_PLBvem.SPLBcem(data_view, Z_init, Z_init,labels,K)
            end_time = time.time()


            phiK = np.asarray(part)

            time_ = end_time - start_time
            acc = np.around(accuracy(labels, phiK), 3)
            nmi = np.around(normalized_mutual_info_score(labels, phiK), 3)
            ari = np.around(adjusted_rand_score(labels, phiK), 3)
            purity = np.around(purity_score(labels, phiK), 3)
            print("Accuracy : ", acc)
            print("nmi : ", nmi)
            print("ari : ", ari)
            print("purity : ", purity)

            df_results.Algorithm[cpt] = 'SPLBM'
            df_results.Slice[cpt] = viewName
            df_results.Dataset[cpt] = bdd_name
            df_results.Time[cpt] = str(time_)
            df_results.ACC[cpt] = str(acc)
            df_results.NMI[cpt] = str(nmi)
            df_results.ARI[cpt] = str(ari)
            df_results.Purity[cpt] = str(purity)
            cpt = cpt + 1

    df_results.to_csv(path_to_save +nom_BDD[nb]+ "_ResultsAll_SliceClustering.csv", index=False)

df_results.to_csv(path_to_save + "ResultsAll_SliceClustering.csv", index=False)