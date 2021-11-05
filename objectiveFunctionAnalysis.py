# -*- coding: utf-8 -*-

"""
This code allows us to run the analysis of objective function evolution according to the NMI metric
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import random
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 16})

pathResults = './results/ResultTGM/'
list_datasets= ['DBLP1', 'DBLP2' , 'PubMed_Diabets', 'classic3', 'classic4' , 'ag_news']
NameDatasets = ['DBLP1', 'DBLP2' , 'PubMed-Diabets', 'Classic3', 'Classic4' , 'AG-news']
list_nameFileMod= './results/ResultTGM/Modularity/resultMod_'

configName = 'BOW_Bert_SentenceRo_GLOVE_Entity_'

tableClustering =  pd.read_csv(pathResults+'Results_TGM_'+ configName + '.csv')


listCorr = []
K=2
L=3
fig, axs = plt.subplots(K, L)
fig.set_size_inches(14,10)
cptR = 0
cptC = 0
xlabel_ = ''
topRun = 30
for nb in range(len(list_datasets)):

    bdd_name = list_datasets[nb]
    bdd_name2 = NameDatasets[nb]

    tab_dataset_clustering = tableClustering.loc[tableClustering['Dataset'] == bdd_name]
    tab_dataset_clustering.reset_index(drop=True)
    nmi_vector = np.asarray(tab_dataset_clustering['NMI'])
    ari_vector = np.asarray(tab_dataset_clustering['ARI'])
    pur_vector = np.asarray(tab_dataset_clustering['Purity'])

    dataMod  = np.load(list_nameFileMod + configName + bdd_name +'.npz')
    listModularite = np.asarray(dataMod['a'])

    corr_ ,_= pearsonr(listModularite,nmi_vector)
    listCorr.append(corr_)
    #corr_ = np.around(corr_,2)

    indexMod = np.argsort(listModularite)
    listModularityNorm = (listModularite - np.min(listModularite)) / (np.max(listModularite) - np.min(listModularite));

    x = np.arange(len(indexMod))
    lns1 = axs[cptR, cptC].plot(x[-topRun:],listModularityNorm[indexMod[-topRun:]],'o--',label='Objective function',c='b')
    ax2 = axs[cptR, cptC].twinx()
    lns2 =ax2.plot(x[-topRun:], np.around(nmi_vector[indexMod[-topRun:]],2), 'o--', label='NMI',c='g')

    if cptC == (L-1):
        ax2.set_ylabel('NMI', size=20)
    #ax2.legend()
    #axs[cptR, cptC].plot(x, nmi_vector[indexMod], 'o--', label='NMI')
    #axs[cptR, cptC].plot(x, pur_vector[indexMod], 'o--', label='Purity')
    #axs[cptR, cptC].plot(x, ari_vector[indexMod], 'o--', label='ARI')
    if cptC==0:
        axs[cptR, cptC].set_ylabel('Normalized objective function')
    axs[cptR, cptC].set_xlabel(xlabel_)
    #axs[cptR, cptC].set_ylabel('Normalized objective function')
    #axs[cptR, cptC].legend()
    axs[cptR, cptC].set_title(bdd_name2 + '('+str(corr_)[0:4]+')' )

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    axs[cptR, cptC].legend(lns, labs, loc=0)

    cptC = cptC +1
    if cptC == L:
        cptC =0
        cptR= cptR+1
        xlabel_ = 'Runs'

my_dpi = 100
plt.tight_layout()
plt.savefig(pathResults+'Of_vs_NMI_alldatasets.pdf')
plt.show()

