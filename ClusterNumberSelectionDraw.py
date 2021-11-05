# -*- coding: utf-8 -*-

"""
This code allows us to plot the results of the cluster number selection
"""
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 16})
path_to_save = './results/ResultTGM/ClusterNumberSelection/'

datasets = ['DBLP1','DBLP2','PubMed_Diabets' ,'classic3', 'classic4', 'ag_news']
datasetsName = ['DBLP1','DBLP2','PubMed-Diabets' ,'Classic3', 'Classic4', 'AG-news']

gold_n = [3,3,3,3, 4, 4]

config = [['BOW', 'Bert', 'SentenceRo', 'GLOVE', 'Entity']]

K=2
L=3
fig, axs = plt.subplots(K, L)
fig.set_size_inches(12,8)
cptR = 0
cptC = 0
for c in range(len(config)):
    slices = config[c]
    subname = ''
    for s in range(len(slices)):
        subname = subname + slices[s] + '_'


    file_to_read = path_to_save + "Results_TGM_" + subname + ".csv"

    df = pd.read_csv(file_to_read)

    i_c =0
    for i, dataset in enumerate(datasets):
        data = df.loc[df['Dataset'] == dataset]
        data['Modularity'] = (data['Modularity'] - data['Modularity'].min()) / (data['Modularity'].max() -
                                                                                data['Modularity'].min())

        res = data.groupby(['ClusterNum'], as_index=False)['Modularity'].quantile(q=0.75)

        axs[cptR, cptC].plot(res['ClusterNum'], res['Modularity'], 'o--',c='b')
        axs[cptR, cptC].vlines(gold_n[i_c], ymin=0, ymax=1, colors='red', linestyles='dashed')
        if cptR ==1:
            axs[cptR, cptC].set_xlabel("Number of clusters")
        if cptC == 0:
            axs[cptR, cptC].set_ylabel("Normalized Objective function")
        axs[cptR, cptC].set_title(datasetsName[i])
        cptC = cptC + 1
        if cptC == L:
            cptC = 0
            cptR = cptR + 1

        i_c += 1

        #plt.grid()
    plt.tight_layout()
    plt.savefig(path_to_save+'ClusterNumberSelection' + dataset + '_normalized.pdf')

    plt.show()

