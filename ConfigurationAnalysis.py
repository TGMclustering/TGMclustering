# -*- coding: utf-8 -*-

"""
This code allows us to run the configuration slices analysis for the TGM model
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 18})
pathResults = './results/ResultTGM/'
path_to_graph = './results/resultSliceClustering/'
list_datasets= ['DBLP1', 'DBLP2' , 'PubMed_Diabets', 'classic3','classic4' , 'ag_news']
NameDatasets = ['DBLP1', 'DBLP2' , 'PubMed-Diabets', 'Classic3', 'Classic4', 'AG-news']

config = [['BOW', 'SentenceRo','Entity'], ['BOW', 'SentenceRo','Entity'],['BOW', 'Bert', 'SentenceRo', 'GLOVE', 'Entity'],
          ['BOW', 'Bert', 'GLOVE', 'Entity'],['BOW', 'Bert', 'Entity'], ['BOW', 'SentenceRo', 'Entity'],
          ['BOW', 'Bert'], ['BOW', 'SentenceRo'], ['Bert', 'Entity'],
          ['SentenceRo', 'Entity'],['BOW', 'SentenceRo', 'GLOVE', 'Entity'],['BOW', 'SentenceRo', 'Bert', 'Entity'],
          ['BOW', 'GLOVE', 'Entity'],['SentenceRo', 'GLOVE', 'Entity']]

tabVide = pd.DataFrame(columns=["Dataset", "Time", "ACC", "NMI", "ARI", "Purity",'Config'])
for c , configName in enumerate(config):
    slices = config[c]
    configName = ''
    for s in range(len(slices)):
        configName = configName + slices[s] + '_'
    if len(slices)==5:
        configName2= 'All'
    else:
        configName2 = configName


    tableClustering =  pd.read_csv(pathResults+'Results_TGM_'+ configName + '.csv')


    tableClustering['Config'] = configName[:-1] .replace("_", "&")
    tableClustering['SliceNb'] =len(slices)
    if len(slices)< 4:
        tableClustering['Slice']  = str(len(slices)) + 'Slices'
    else:
        tableClustering['Slice'] =   '> 4 Slices'
    print('tableClustering ', tableClustering.shape)
    tabVide = pd.concat([tabVide, tableClustering], axis=0)

print('tabVide ', tabVide.columns.values)
print(tabVide.head())
tab_all_graph = pd.read_csv(path_to_graph+'ResultsAll_SliceClustering.csv')
tab_all_graph_CoclustMod = tab_all_graph.loc[(tab_all_graph['Algorithm'] == 'CoclustMod')]
tab_all_graph_CoclustMod = tab_all_graph_CoclustMod.reset_index(drop=True)
tab_all_graph_CoclustMod = tab_all_graph_CoclustMod[['Dataset', 'Time', 'ACC' ,'NMI' ,'ARI' ,'Purity', 'Slice']]
tab_all_graph_CoclustMod.columns   = ['Dataset', 'Time', 'ACC' ,'NMI' ,'ARI' ,'Purity', 'Config']
tab_all_graph_CoclustMod['SliceNb']=1
tab_all_graph_CoclustMod['Slice']='1Slices'
print('tab_all_graph', tab_all_graph.columns.values)

uniqueConfigSlice = tabVide[['Config','Slice','SliceNb']].drop_duplicates()
configVectore  = np.unique(np.asarray(tabVide['Config']))
K=2
L=3
fig, axs = plt.subplots(K, L)
cptR = 0
cptC = 0
xlabel_ = ''
topRun = 10


colorVector = ['#5c85c0','#82654b','#56448b','#e05c70','#64b2bf','#a78a7c','#6e68ef','#ed91dd','#80b7fd','#9d9f96','#926eb6','#f1afaa']

fig.set_size_inches(16,10)
for nb in range(len(list_datasets)):

    bdd_name = list_datasets[nb]
    bdd_name2 = NameDatasets[nb]
    tab_dataset_clustering = tabVide.loc[(tabVide['Dataset'] == bdd_name)]
    print('tab_dataset_clustering ', tab_dataset_clustering.shape)
    tab_dataset_clustering =tab_dataset_clustering.reset_index(drop=True)
    print('tab_dataset_clustering apres ', tab_dataset_clustering.shape)


    nmi_vector = np.asarray(tab_dataset_clustering['NMI'])
    ari_vector = np.asarray(tab_dataset_clustering['ARI'])
    pur_vector = np.asarray(tab_dataset_clustering['Purity'])
    configVectore  = np.asarray(tab_dataset_clustering['Config'])
    print('configVectore ',np.unique(configVectore))
    nmiPuritzMedian = tab_dataset_clustering.groupby(tab_dataset_clustering.Config)[['NMI','Purity']].median()

    print('nmiPuritzMedian ', nmiPuritzMedian)
    nmi_vectorMedian = np.asarray(nmiPuritzMedian['NMI'])
    pur_vectorMedian = np.asarray(nmiPuritzMedian['Purity'])

    indexVecrtor = list(nmiPuritzMedian.index)
    print(indexVecrtor)

    nmiPuritzMedian['Config']=indexVecrtor
    nmiPuritzMedian = nmiPuritzMedian.reset_index(drop=True)
    print(nmiPuritzMedian.head())



    uniqueConfigSliceJoin = pd.merge(uniqueConfigSlice, nmiPuritzMedian, on='Config', how='left')

    print('uniqueConfigSliceJoin', uniqueConfigSliceJoin.shape)

    uniqueConfigSliceJoinSort =uniqueConfigSliceJoin.sort_values(by=['SliceNb','NMI'],ascending=False)
    uniqueConfigSliceJoinSort =uniqueConfigSliceJoinSort.reset_index(drop=True)

    nmi_vectorMedian = np.asarray(uniqueConfigSliceJoinSort['NMI'])
    pur_vectorMedian = np.asarray(uniqueConfigSliceJoinSort['Purity'])
    indexVecrtor =  np.asarray(uniqueConfigSliceJoinSort['Config'])
    indexVecrtor[indexVecrtor=='BOW&Bert&SentenceRo&GLOVE&Entity'] = 'ALL'
    nbSlices_vector = np.asarray(uniqueConfigSliceJoinSort['SliceNb'])
    listIndexBar = []
    indBegin = 0
    nbSliceType = [4,3,2]
    for ns, nsNum in enumerate(nbSliceType):
        indBegin = indBegin+1
        listIndexCurrent = []
        for ind,e in enumerate(nbSlices_vector):
            if ns==0:
                if e >= nsNum:
                    listIndexCurrent.append(indBegin)
                    indBegin = indBegin+1
            else:
                if ((e  >= nsNum) &  (e < nbSliceType[ns-1])):
                    listIndexCurrent.append(indBegin)
                    indBegin = indBegin+1

        listIndexBar.append(listIndexCurrent)

    print('listIndexBar', len(listIndexBar))
    indexXtickLabel = []
    prec=1
    for le in range(len(listIndexBar)):
        lenghtSubList = len(listIndexBar[le])
        indexCurrentLabel= lenghtSubList / 2
        indexXtickLabel.append(prec+indexCurrentLabel)
        prec = prec+ 2 + indexCurrentLabel

    listIndexBararray = list(matplotlib.cbook.flatten(listIndexBar))

    print(colorVector)
    bars = axs[cptR, cptC].bar(listIndexBararray,nmi_vectorMedian, color=colorVector, label=indexVecrtor,width = 0.8,align='center')
    axs[cptR, cptC].axhline(y=np.amax(nmi_vectorMedian), color='r', linestyle='--')
    axs[cptR, cptC].set_xticks(indexXtickLabel)
    axs[cptR, cptC].set_xticklabels(['$\geq$4 Slices', '3 Slices','2 Slices'])


    colors = dict(zip(indexVecrtor, colorVector))
    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]

    axs[cptR, cptC].set_title(bdd_name2, size=16)

    if cptC==0:
        axs[cptR, cptC].set_ylabel('NMI', size = 20)
    cptC = cptC +1
    if cptC == L:
        cptC =0
        cptR= cptR+1
        xlabel_ = 'Runs'

my_dpi=100
plt.tight_layout()
plt.legend(handles, labels,  loc = (-1.9,2.3), ncol=3)
plt.subplots_adjust( top=0.8)
#plt.subplots_adjust( bottom=0.05, hspace=0.13)

plt.savefig(pathResults+'configComprison.pdf')#, dpi=my_dpi)
plt.show()
