# -*- coding: utf-8 -*-

"""
This code allows us to run classical clustering approaches namely Kmeans, Spherical Kmeans and Auto-encoder
"""

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from coclust import  clustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from coclust.evaluation.external import accuracy
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics
import time
import random
from keras.models import Model
from keras.layers import Dense, Input


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
path_to_save = './results/resultViewClustering/'
nom_BDD = ['DBLP1','DBLP2','PubMed_Diabets' ,'classic3', 'classic4', 'ag_news']

nCLusters = [3,3,3,3, 4,  4]

nbrIteration = 30

nbSlices = 8
nbAlgorithms = 3
df_results = pd.DataFrame(columns=["Dataset", "Time", "ACC", "NMI", "ARI", "Purity",'Algorithm','View'],
                          index=np.arange(nbrIteration * ((len(nom_BDD) * nbSlices*nbAlgorithms ))).tolist())
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
    simBow = np.load(global_path +bdd_name+'/' + 'view_bow' + '.npz')
    simBow = simBow['arr_0']
    print('simBow ', simBow.shape)

    simBert = np.load(global_path +bdd_name+'/' + 'view_bert-base-cased' + '.npz')
    simBert = simBert['arr_0']

    simBertLPCA = np.load(global_path +bdd_name+'/' + 'view_avgpca__bert-large-cased' + '.npz')
    simBertLPCA = simBertLPCA['arr_0']
    print('simBertLPCA ', simBertLPCA.shape)

    simRoBertLPCA = np.load(global_path +bdd_name+'/' + 'view_avgpca__roberta-large' + '.npz')
    simRoBertLPCA = simRoBertLPCA['arr_0']

    simRoBerta = np.load(global_path +bdd_name+'/' + 'view_roberta-large' + '.npz')
    simRoBerta = simRoBerta['arr_0']

    simSentenceRoBerta = np.load(global_path +bdd_name+'/'+ 'sim_sentenceRoberta' +'.npz')
    simSentenceRoBerta = simSentenceRoBerta['arr_0']
    print('simSentenceRoBerta ', simSentenceRoBerta.shape)

    simGlove = np.load(global_path +bdd_name+'/' + 'view_glove' + '.npz')
    simGlove = simGlove['arr_0']
    print('simGlove ', simGlove.shape)

    simW2V = np.load(global_path +bdd_name+'/' + 'view_w2v' + '.npz')
    simW2V = simW2V['arr_0']

    simEntity = np.load(global_path +bdd_name+'/' + 'view_entity' + '.npz')
    simEntity = simEntity['arr_0']
    print('simEntity ', simEntity.shape)

    viewsNames = ['BOW','Bert','BertLPCA','RoBertLPCA', 'SentenceRo', 'GLOVE', 'W2V','Entity']
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
            random.seed(it)
            np.random.seed(it)
            print("iter " + str(it))
            ##########################################################
            print('################### skmeans ######################')
            Z_init = random_init(K, n_new)
            colSum= data_view.sum(0)
            print('colSum',colSum)
            print(np.sum(colSum==0))
            start_time = time.time()
            model = clustering.spherical_kmeans.SphericalKmeans(n_clusters=K,  max_iter=100, n_init=1,tol=1e-09, weighting=False)
            model.fit(data_view)
            end_time = time.time()

            phiK = model.labels_
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

            df_results.Algorithm[cpt] = 'Skmeans'
            df_results.View[cpt] = viewName
            df_results.Dataset[cpt] = bdd_name
            df_results.Time[cpt] = str(time_)
            df_results.ACC[cpt] = str(acc)
            df_results.NMI[cpt] = str(nmi)
            df_results.ARI[cpt] = str(ari)
            df_results.Purity[cpt] = str(purity)
            cpt = cpt + 1


            ##########################################################
            print('################### kmeans ######################')
            Z_init = random_init(K, n_new)

            start_time = time.time()
            kmeans = KMeans(n_clusters=K, random_state=0).fit(data_view)

            end_time = time.time()

            phiK = kmeans.labels_
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

            df_results.Algorithm[cpt] = 'Kmeans'
            df_results.View[cpt] = viewName
            df_results.Dataset[cpt] = bdd_name
            df_results.Time[cpt] = str(time_)
            df_results.ACC[cpt] = str(acc)
            df_results.NMI[cpt] = str(nmi)
            df_results.ARI[cpt] = str(ari)
            df_results.Purity[cpt] = str(purity)
            cpt = cpt + 1

            ##########################################################
            ##########################################################
            print('################### GMM ######################')
            Z_init = random_init(K, n_new)

            start_time = time.time()
            gmm = GaussianMixture(n_components=K, covariance_type='diag',reg_covar=1e-5)
            phiK = gmm.fit_predict(data_view)
            end_time = time.time()


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

            df_results.Algorithm[cpt] = 'GMM'
            df_results.View[cpt] = viewName
            df_results.Dataset[cpt] = bdd_name
            df_results.Time[cpt] = str(time_)
            df_results.ACC[cpt] = str(acc)
            df_results.NMI[cpt] = str(nmi)
            df_results.ARI[cpt] = str(ari)
            df_results.Purity[cpt] = str(purity)
            cpt = cpt + 1

            ##########################################################
            print('################### Auto-encoder ######################')
            Z_init = random_init(K, n_new)
            colSum= data_view.sum(0)
            print('colSum',colSum)
            print(np.sum(colSum==0))

            inputs_dim = data_view.shape[1]
            encoder = Input(shape=(inputs_dim,))
            e = Dense(1024, activation="relu")(encoder)
            e = Dense(512, activation="relu")(e)
            e = Dense(256, activation="relu")(e)
            ## bottleneck layer
            n_bottleneck = 15
            ## defining it with a name to extract it later
            bottleneck_layer = "bottleneck_layer"
            # can also be defined with an activation function, relu for instance
            bottleneck = Dense(n_bottleneck, name=bottleneck_layer)(e)
            ## define the decoder (in reverse)
            decoder = Dense(256, activation="relu")(bottleneck)
            decoder = Dense(512, activation="relu")(decoder)
            decoder = Dense(1024, activation="relu")(decoder)
            ## output layer
            output = Dense(inputs_dim)(decoder)
            ## model
            model = Model(inputs=encoder, outputs=output)
            model.summary()

            start_time = time.time()
            encoder = Model(inputs=model.input, outputs=bottleneck)
            model.compile(loss="mse", optimizer="adam")
            history = model.fit(
                data_view,
                data_view,
                batch_size=32,
                epochs=25,
                verbose=1
            )

            data_encoded = encoder.predict(data_view)

            kmeans = KMeans(n_clusters=K, random_state=0).fit(data_encoded)
            end_time = time.time()


            phiK = kmeans.labels_
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

            df_results.Algorithm[cpt] = 'Autoencoder'
            df_results.View[cpt] = viewName
            df_results.Dataset[cpt] = bdd_name
            df_results.Time[cpt] = str(time_)
            df_results.ACC[cpt] = str(acc)
            df_results.NMI[cpt] = str(nmi)
            df_results.ARI[cpt] = str(ari)
            df_results.Purity[cpt] = str(purity)
            cpt = cpt + 1

    df_results.to_csv(path_to_save + "ResultsAll_ViewClustering_"+bdd_name+".csv", index=False)

df_results.to_csv(path_to_save + "ResultsAll_ViewClustering.csv", index=False)

##################################################################
#                        Entity recogination                     #
##################################################################
listEntity = []
for a, abs in enumerate(abstracts_cleaned):
    doc = nlp(str(abs))
    listInter = []
    # returns all entities in the whole document
    all_linked_entities = doc._.linkedEntities
    # iterates over sentences and prints linked entities
    for sent in doc.sents:
        # sent._.linkedEntities.pretty_print()

        for i in range(len(sent._.linkedEntities)):
            # print(sent._.linkedEntities[0].get_id())
            entity = sent._.linkedEntities[i].get_id()
            listInter.append(entity)
    listEntity.append(listInter)

listEntityAll = functools.reduce(operator.concat, listEntity)
print("listEntityAll ", len(listEntityAll))
listEntityAllUnique = np.unique(np.asarray(listEntityAll))
print("listEntityAllUnique ", len(listEntityAllUnique))
dictIndex = dict(zip(listEntityAllUnique, np.arange(len(listEntityAllUnique))))
entityEmbeddings = np.zeros((len(abstracts), len(listEntityAllUnique)))
for a in range(len(abstracts)):
    lista = listEntity[a]
    indexEn = values = np.asanyarray(list(map(dictIndex.get, lista)))
    print(indexEn)
    entityEmbeddings[a, indexEn] = 1