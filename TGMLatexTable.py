# -*- coding: utf-8 -*-

"""
This code allows us to generate the latex table of all results
"""
import numpy as np
import pandas as pd
from pylab import *

###################################################################################################
###################################################################################################
global_path = './data/'
path_to_save = './ResultTGM/'
path_to_clustring ='./results/resultViewClustering/'
path_to_graph = './results/resultSliceClustering/'
path_to_TensorDecomp = './results/resultTensorDecomp/'
path_to_TSPLBM = './results/ResultTSPLBM/'
nom_BDD = ['DBLP1','DBLP2','PubMed_Diabets' , 'classic3','classic4', 'ag_news']
NameDatasets = ['DBLP1', 'DBLP2' , 'PubMed_Diabets','Classic3', 'Classic4', 'AG-news']

nCLusters = [3,3,3,3, 4, 4]

config =[['BOW','Bert', 'SentenceRo','GLOVE', 'Entity']]

noms_couches = ['BOW', 'BertLPCA', 'SentenceRo', 'GLOVE', 'Entity']
list_metrics = [ 'Purity']#, 'ARI', 'NMI'

All_text_latex = ''
listDf = []
for c, conf in enumerate(config):
    slices = config[c]
    subname =''
    for s in range(len(slices)):
        subname= subname + slices[s]+'_'

    tab_TGM = pd.read_csv(path_to_save + "Results_TGM_"+subname+".csv")
    listDf.append(tab_TGM)


tab_all_clustering = pd.read_csv(path_to_clustring+'ResultsAll_ViewClustering.csv')
tab_autoEncoder_clustering = pd.read_csv(path_to_clustring+"ResultsAll_ViewClustering_LDA.csv")
tab_all_clustering = pd.concat([tab_all_clustering,tab_autoEncoder_clustering], axis=0)
tab_all_graph = pd.read_csv(path_to_graph+'ResultsAll_SliceClustering.csv')

tab_all_tensorDecomp = pd.read_csv(path_to_TensorDecomp+'ResultsEntity_TensorDecomp.csv')
tab_tsplbm           = pd.read_csv(path_to_TSPLBM+"Results_TSPLBM_"+subname+".csv")
tab_tsplbm['Algorithm'] = 'TSPLBM'
tab_all_tensorDecomp = pd.concat([tab_all_tensorDecomp,tab_tsplbm], axis=0)
for nb in range(len(nom_BDD)):

    bdd_name = nom_BDD[nb]
    bdd_name2 = NameDatasets[nb]
    print('#################### BDD : ' + nom_BDD[nb] + ' ##########################')
    debut_latex = '\\multirow{6}{*}{\\rotatebox[origin=c]{90}{\\tt ' + bdd_name2 + '}}'
    text_Kmeans = ' &  \\multirow{3}{*}{\\rotatebox[origin=c]{0}{\\tt Clustering}} &   K-means &  '
    text_SKmeans = ' & &   Spherical K-means  & '
    text_GMM = ' &  & GMM  & '
    text_CoclustInfo = '& \\multirow{3}{*}{\\rotatebox[origin=c]{0}{\\tt Graph clustering}}  &   CoclustInfo  & '
    text_CoclustMod = ' & &  CoclustMod  & '
    text_SPLBM = ' & &  SPLBM    & '

    listTGMtext = []
    for c, conf in enumerate(config):
        slices = config[c]
        subname = ''
        for s in range(len(slices)):
            sliceName = slices[s]
            if s == (len(slices)-1):
                subname = subname + sliceName[0:2]
            else:
                subname = subname + sliceName[0:2]+'-'

        text_TGM = '&  &  $TGM_{' + subname+ '}$ & '
        listTGMtext.append(text_TGM)

    text_parafac = '& \\multirow{'+str(2+len(listTGMtext))+'}{*}{\\rotatebox[origin=c]{0}{\\tt Tensor clustering}} &  PARAFAC    & '
    text_tucker = '& &  Tucker decomp    & '
    text_tsplbm = '& &  TSPLBM    & '

    fin_latex = ' \\hline \\hline '

    saut_ligne_ = ' \\cline{2-' + str((len(noms_couches)*len(list_metrics))+3) + '} '
    saut_ligne = ' \\cline{3-' + str((len(noms_couches)*len(list_metrics))+3) + '} '
    ######################################################
    ######################################################
    # Affichage



    tab_dataset_clustering = tab_all_clustering.loc[tab_all_clustering['Dataset'] == bdd_name]
    tab_dataset_graph = tab_all_graph.loc[tab_all_graph['Dataset'] == bdd_name]
    tab_dataset_tensorDecomp = tab_all_tensorDecomp.loc[tab_all_tensorDecomp['Dataset'] == bdd_name]

    for m, metric in enumerate(list_metrics):
        for c in range(len(listDf)):
            tab_all_TGM = listDf[c]
            tab_dataset_TGM = tab_all_TGM.loc[tab_all_TGM['Dataset'] == bdd_name]
            value_tgm = np.around(np.mean(np.asarray(tab_dataset_TGM[metric])), 2)
            var_tgm = np.around(np.std(np.asarray(tab_dataset_TGM[metric])), 2)
            listTGMtext[c] = listTGMtext[c] + ' \\multicolumn{5}{c|}{' + str(value_tgm) + ' $\pm$ ' + str(var_tgm) + '} &'

        tab_dataset_tensorDecomp_select = tab_dataset_tensorDecomp[tab_dataset_tensorDecomp['Algorithm'] == 'Parafac']
        value_parafac = np.around(np.mean(np.asarray(tab_dataset_tensorDecomp_select[metric])), 2)
        var_parafac = np.around(np.std(np.asarray(tab_dataset_tensorDecomp_select[metric])), 2)
        text_parafac = text_parafac + ' \\multicolumn{5}{c|}{' + str(value_parafac) + ' $\pm$ ' + str(
            var_parafac) + '} &'

        tab_dataset_tensorDecomp_select = tab_dataset_tensorDecomp[tab_dataset_tensorDecomp['Algorithm'] == 'Tucker']
        value_tucker = np.around(np.mean(np.asarray(tab_dataset_tensorDecomp_select[metric])), 2)
        var_tucker = np.around(np.std(np.asarray(tab_dataset_tensorDecomp_select[metric])), 2)
        text_tucker = text_tucker + ' \\multicolumn{5}{c|}{' + str(value_tucker) + ' $\pm$ ' + str(var_tucker) + '} &'

        tab_dataset_tensorDecomp_select = tab_dataset_tensorDecomp[tab_dataset_tensorDecomp['Algorithm'] == 'TSPLBM']
        value_tsplbm = np.around(np.mean(np.asarray(tab_dataset_tensorDecomp_select[metric])), 2)
        var_tsplbm = np.around(np.std(np.asarray(tab_dataset_tensorDecomp_select[metric])), 2)
        text_tsplbm = text_tsplbm + ' \\multicolumn{5}{c|}{' + str(value_tsplbm) + ' $\pm$ ' + str(var_tsplbm) + '} &'
    # nettoyage

    text_parafac = text_parafac[:-1] + ' \\\\'
    text_tucker = text_tucker[:-1] + ' \\\\'
    text_tsplbm = text_tsplbm[:-1] + ' \\\\'
    text_TGM = text_TGM[:-1] + ' \\\\'

    for m, metric in enumerate(list_metrics):
        for c, couche in enumerate(noms_couches):
            ####################################################################
            ####################################################################
            tab_dataset_clustering_slect = tab_dataset_clustering[
                ((tab_dataset_clustering['Algorithm'] == 'Kmeans') & (tab_dataset_clustering['View'] == couche))]
            value_Kmeans = np.around(np.mean(np.asarray(tab_dataset_clustering_slect[metric])), 2)
            var_Kmeans = np.around(np.std(np.asarray(tab_dataset_clustering_slect[metric])), 2)
            text_Kmeans = text_Kmeans + str(value_Kmeans) + ' $\pm$ ' + str(var_Kmeans) + ' & '

            tab_dataset_clustering_slect = tab_dataset_clustering[
                ((tab_dataset_clustering['Algorithm'] == 'Skmeans') & (tab_dataset_clustering['View'] == couche))]
            value_SKmeans = np.around(np.mean(np.asarray(tab_dataset_clustering_slect[metric])), 2)
            var_SKmeans = np.around(np.std(np.asarray(tab_dataset_clustering_slect[metric])), 2)
            text_SKmeans = text_SKmeans + str(value_SKmeans) + ' $\pm$ ' + str(var_SKmeans) + ' & '

            tab_dataset_clustering_slect = tab_dataset_clustering[
                ((tab_dataset_clustering['Algorithm'] == 'Autoencoder') & (tab_dataset_clustering['View'] == couche))]
            value_GMM = np.around(np.mean(np.asarray(tab_dataset_clustering_slect[metric])), 2)
            var_GMM = np.around(np.std(np.asarray(tab_dataset_clustering_slect[metric])), 2)
            text_GMM = text_GMM + str(value_GMM) + ' $\pm$ ' + str(var_GMM) + ' & '

            ####################################################################
            ####################################################################
            tab_dataset_graph_select = tab_dataset_graph[
                ((tab_dataset_graph['Algorithm'] == 'CoclustInfo') & (tab_dataset_graph['Slice'] == couche))]
            value_CoclustInfo = np.around(np.mean(np.asarray(tab_dataset_graph_select[metric])), 2)
            var_CoclustInfo = np.around(np.std(np.asarray(tab_dataset_graph_select[metric])), 2)
            text_CoclustInfo = text_CoclustInfo + str(value_CoclustInfo) + ' $\pm$ ' + str(var_CoclustInfo) + ' & '

            tab_dataset_graph_select = tab_dataset_graph[
                ((tab_dataset_graph['Algorithm'] == 'CoclustMod') & (tab_dataset_graph['Slice'] == couche))]
            value_CoclustMod = np.around(np.mean(np.asarray(tab_dataset_graph_select[metric])), 2)
            var_CoclustMod = np.around(np.std(np.asarray(tab_dataset_graph_select[metric])), 2)
            text_CoclustMod = text_CoclustMod + str(value_CoclustMod) + ' $\pm$ ' + str(var_CoclustMod) + ' & '

            tab_dataset_graph_select = tab_dataset_graph[
                ((tab_dataset_graph['Algorithm'] == 'SPLBM') & (tab_dataset_graph['Slice'] == couche))]
            value_SPLBM = np.around(np.mean(np.asarray(tab_dataset_graph_select[metric])), 2)
            var_SPLBM = np.around(np.std(np.asarray(tab_dataset_graph_select[metric])), 2)
            text_SPLBM = text_SPLBM + str(value_SPLBM) + ' $\pm$ ' + str(var_SPLBM) + ' & '

            ####################################################################
            ####################################################################
    text_Kmeans = text_Kmeans[:-2] + ' \\\\'
    text_SKmeans = text_SKmeans[:-2] + ' \\\\'
    text_GMM = text_GMM[:-2] + ' \\\\'
    text_CoclustInfo = text_CoclustInfo[:-2] + ' \\\\'
    text_CoclustMod = text_CoclustMod[:-2] + ' \\\\'
    text_SPLBM = text_SPLBM[:-2] + ' \\\\'

    textfinale = text_Kmeans + saut_ligne + text_SKmeans + saut_ligne + text_GMM + saut_ligne_ + text_CoclustInfo + saut_ligne + text_CoclustMod + saut_ligne + text_SPLBM + saut_ligne_ + text_parafac + saut_ligne + text_tucker + saut_ligne + text_tsplbm
    for c in range(len(listTGMtext)):
        textfinale = textfinale + saut_ligne + listTGMtext[c]


    All_text_latex = All_text_latex + debut_latex + textfinale + fin_latex

file = open(path_to_save+"TableauCompraisonTGM.txt", "w")
file.write(All_text_latex)
