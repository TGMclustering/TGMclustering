# -*- coding: utf-8 -*-

"""TGM consensius

This code allow to perfom consensus analysis and generate the results that compares
TGM results with and without consensus

You need to install

pip install Cluster_Ensembles

!sudo apt-get install metis

"""




import numpy as np
import pandas as pd 
from sklearn import metrics
import Cluster_Ensembles as CE
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)




list_datasets= ['DBLP1', 'DBLP2' , 'PubMed_Diabets' ,'classic3', 'classic4','ag_news']

config = [['BOW', 'Bert', 'SentenceRo', 'GLOVE', 'Entity']]


path_to_save = './data/'
path_to_saveTGM = './results/ResultTGM/'


All_text_latex =''
nCLusters = [3,3,3,3, 4,  4]
topRun =30
for nb in range(len(list_datasets)):

    print('#################### BDD : ' + list_datasets[nb] + ' ##########################')
    # read the true labels of the dataset
    bdd_name = list_datasets[nb]
    inf_doc = pd.read_csv(path_to_save+ bdd_name+'/'+ bdd_name + '.csv', delimiter=',')
    # print(inf_doc)
    labels_all = np.asarray(inf_doc['label']).astype(int)
    labels = labels_all.tolist()



    # definir le nombre de clusters 
    K = nCLusters[nb]
    
    print('#################### BDD : ' + list_datasets[nb] + ' ##########################')


    listValueswoConsensus = []
    listValuesConsensus = []
    listConfigNames = []
    for c, conf in enumerate(config):
        slices = config[c]
        subname = ''
        subname2 = ''
        for s in range(len(slices)):

            sliceName = slices[s]
            subname = subname + sliceName + '_'
            if s == (len(slices)-1):
                subname2 = subname2 + sliceName[0:2]
            else:
                subname2 = subname2 + sliceName[0:2]+'-'
        # lire le fichier resultats pour w/o consensus

        tab_all_TGM = pd.read_csv(path_to_saveTGM + "Results_TGM_"+subname+".csv")


        listConfigNames.append(subname2)

        # selectionner la base 
        tab_dataset_TGM = tab_all_TGM.loc[tab_all_TGM['Dataset'] == bdd_name]
        # calcul de la moyenne 
        value_tgm = np.around(np.mean(np.asarray(tab_dataset_TGM['NMI'])), 2)

        # mettre le resultat dans une liste 
        listValueswoConsensus.append(value_tgm)
        # lire le fichier resultats pour consensus
        dataLoad = np.load(path_to_saveTGM+'/Partition/resultCLusteringPartion_' + subname + '_' + bdd_name+'.npz')
        cluster_runs = np.asarray(dataLoad['a'])

        dataMod  = np.load(path_to_saveTGM+'/Modularity/resultMod_' + subname + bdd_name +'.npz')
        listModularite = np.asarray(dataMod['a'])
        indexMod = np.argsort(listModularite)
        indexTopRuns = indexMod[-topRun:]

   
        # calaculer le consensus 
        consensus_clustering_labels = CE.cluster_ensembles(cluster_runs[indexTopRuns,:], verbose = True, N_clusters_max = K) 
        # calcul NMI, Purity
        nmi = np.around(normalized_mutual_info_score(labels, consensus_clustering_labels), 2)
        # mettre les resultats dans une liste 
        listValuesConsensus.append(nmi)


    listValueswoConsensus = np.asarray(listValueswoConsensus)
    listValuesConsensus   = np.asarray(listValuesConsensus)
    listConfigNames       = np.asarray(listConfigNames)
    # trouver le meilleur resultats et le meilleurs modeles w/o consensus
    indexBestwoConsensus = np.argmax(listValueswoConsensus)
    bestValuewoConsensus = listValueswoConsensus[indexBestwoConsensus]
    bestCOnfigwoConsensus = listConfigNames[indexBestwoConsensus]
    # trouver le meilleur resultats et le meilleurs modeles consensus
    
    indexBestConsensus = np.argmax(listValuesConsensus)
    bestValueConsensus = listValuesConsensus[indexBestConsensus]
    bestCOnfigConsensus = listConfigNames[indexBestConsensus]

    bestValueConsensusConfiConsensus = listValuesConsensus[indexBestConsensus]
    bestCOnfigConsensusConfiConsensus = listConfigNames[indexBestConsensus]

    bestValueConsensusConfiwoConsensus = listValuesConsensus[indexBestwoConsensus]
    bestCOnfigConsensusConfiwoConsensus = listConfigNames[indexBestwoConsensus]
    
    # calculer le pourcentage d'improvement 
    improveConfiwoConsensus = np.around(((bestValueConsensusConfiwoConsensus - bestValuewoConsensus)/bestValuewoConsensus)*100,2)
    improveConfiConsensus = np.around(((bestValueConsensusConfiConsensus - bestValuewoConsensus)/bestValuewoConsensus)*100,2)

    # rajouter la valeur dans le tableau 
    ligneLatex =  bdd_name + '&' + bestCOnfigwoConsensus  + '&' + str(bestValuewoConsensus) + '&' + str(bestValueConsensusConfiwoConsensus) + '&'+ str(improveConfiwoConsensus) + '&' + bestCOnfigConsensus + '&' + str(bestValueConsensus) + '&' + str(improveConfiConsensus) + '\\\\' + '\\hline '
    # mettre dans le tableau latex 
    All_text_latex = All_text_latex + ligneLatex

# enregistemenet du tableau finale 


file = open("TableauConsensusTGM.txt", "w")
file.write(All_text_latex)

All_text_latex

file = open("TableauConsensusTGM.txt", "w")
file.write(All_text_latex)