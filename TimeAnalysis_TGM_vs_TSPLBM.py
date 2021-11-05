
# -*- coding: utf-8 -*-

"""
This code allows us to generate the plot of time analysis comparing TGM to TSPLBM algorithm
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 16})
global_path = './data/'
pathResults = './results/ResultTGM/'
path_to_save_TSPLBM = './results/ResultTSPLBM/'
path_to_save_TGM= './results/ResultTGM/'

nom_BDD = ['DBLP1','DBLP2','PubMed_Diabets' ,'classic3', 'classic4', 'ag_news']
nom_BDD2 = ['DBLP1','DBLP2','PubMed-Diabets' ,'Classic3', 'Classic4', 'AG-news']

nCLusters = [3,3,3,3, 4,  4]


nbrIteration = 30
config = [['BOW', 'Bert', 'SentenceRo', 'GLOVE', 'Entity']]

slices = config[0]
subname = ''
for s in range(len(slices)):
    subname = subname + slices[s] + '_'

result_table_TGM = pd.read_csv(path_to_save_TGM+ "Results_TGM_"+subname+".csv")
result_table_TGM['Algorithm'] ='TGM'
result_table_TSPLBM = pd.read_csv(path_to_save_TSPLBM+ "Results_TSPLBM_"+subname+".csv")
result_table_TSPLBM['Algorithm'] ='TSPLBM'

time_result_table = pd.concat([result_table_TGM,result_table_TSPLBM], axis=0)
time_result_table= time_result_table[time_result_table.Dataset!='bbc']

time_result_table_ = time_result_table.groupby(['Algorithm','Dataset'])[['Time']].median()
time_result_table_n =time_result_table_.reset_index(level=[ 'Algorithm','Dataset'])

tgm_vector =time_result_table_n.Time[time_result_table_n.Algorithm=='TGM']
tsplbm_vector = time_result_table_n.Time[time_result_table_n.Algorithm=='TSPLBM']

fig, ax = plt.subplots()
fig.set_size_inches(9,7.5)
x = np.arange(len(nom_BDD))  # the label locations
width = 0.35  # the width of the bars
rects1 = ax.bar(x - width/2, np.round(tgm_vector), width, label='TGM', color= '#5c85c0' )
rects2 = ax.bar(x + width/2, np.round(tsplbm_vector), width, label='TSPLBM', color= '#e05c70')

ax.set_ylabel('Time(s)')
ax.set_xticks(x)
ax.set_xticklabels(nom_BDD2,rotation=40)

ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
#ax.spines['left'].set_visible(False)

ax.bar_label(rects1, padding=3,color ='#2e4cab',size = 18)
ax.bar_label(rects2, padding=3,color= '#bd2b41',size = 18)

plt.tight_layout()
plt.savefig(pathResults+'timeComparisonTGM_TSPLBM.pdf')#, dpi=my_dpi)
plt.show()

