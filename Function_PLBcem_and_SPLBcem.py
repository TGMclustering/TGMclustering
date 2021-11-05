#!/usr/bin/env python

"""
This code allows us to run SPLBM algorithm
"""
from math import *
import numpy as np
from numpy import *
from pylab import *

import scipy.sparse as sp
from coclust.evaluation.external import accuracy
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.feature_extraction.text import *
from sklearn.metrics.cluster import adjusted_rand_score
import time

#############################input data#####################################################################
"""
####cstr#########################################
mat=loadmat("/users/lipade2/mailem/pylouvain/A_tester/cstr_ori.mat")
m = mat['fea']

labels=mat['gnd']

labels=labels.tolist()
labels = list(itertools.chain.from_iterable(labels))
labels=[l-1 for l in labels]
"""
#################################
def SPLBcem(matrice,Z,W,labels ,K):

    m=matrice
    
  
    
    
    ###########################################################################################################
    
    const=1./(1.*m.sum()*m.sum())
    
    m=sp.csr_matrix(m)
    
    
    
    
    
    N=m.sum()
    
    
    
    #####degres lignes et colonnes
    xi_=sp.lil_matrix(m.sum(axis=1))
    x_j=sp.lil_matrix(m.sum(axis=0))
    
    
    
    nb_rows=m.shape[0]
    nb_cols=m.shape[1]
    
    ###sparsity
    print (1-(m.nnz/(nb_rows*nb_cols*1.)))
    
    
    
    #L=3
    #############################################################################################################################################################################################
    ##########################################################################################Initialisation#####################################################################################
    #############################################################################################################################################################################################
    
    
    #############initialization Z et W###########################################################################
    ##Z
    #Z=np.zeros((nb_rows,K))
    #Z_a=np.random.randint(K,size=nb_rows)
    #Z=np.zeros((nb_rows,K))
    #Z[np.arange(nb_rows) , Z_a]=1
    Z=sp.lil_matrix(Z)
    
    ###W
    #W=np.zeros((nb_cols,K))
    #W_a=np.random.randint(K,size=nb_cols)
    #W=np.zeros((nb_cols,K))
    #W[np.arange(nb_cols) , W_a]=1
    W=sp.lil_matrix(W)
    
    
    #######proportions
    
    ############pik proportion lignes
    
    #n=Z.sum()
    pik=Z.sum(axis=0)
    pik=pik/nb_rows
    
    ############pl proportions colonnes
    #d=W.sum()
    pl=W.sum(axis=0)
    pl=pl/nb_cols
    
    pik=np.squeeze(np.asarray(pik))
    pl=np.squeeze(np.asarray(pl))
    
    
    
    #######gammakk (dans le code jappelle gamma ==> delta)
    
    
    Xw=m*W
    
    Xz=m.T*Z	
    
        
    Xw_k=Xw.sum(axis=0)
    Xz_k=Xz.sum(axis=0)
    
    
    
    den=  Xz_k.T * Xw_k
    diag_den=den.diagonal()
    sumden=diag_den.sum()
    
    
    den=1./(den+const)
    Num= ((Z.T * m) * W)
    deltakk=Num.multiply(den)
    
    
    diag_Num=Num.diagonal()
    sumpkk=np.sum(diag_Num)
    
    
    diago=deltakk.diagonal()
    diago=diago+1
    deltakk=np.diag(np.squeeze(np.asarray(diago)))
    
    delta=(N-sumpkk)/((1.*N*N)-sumden)
    deltakk[deltakk == 0] = delta
    
    ##########################################################################################################################	
    ##############################################################co-clustering###############################################
    ##########################################################################################################################	
    change=True
    news=[]
    Linit=float(-inf)
    
    em=0
    time_start = time.time()
    
    LBCEM=[]
    while change and em<60:
        change=False
    
        
        ##########rows
            
        Xw= m * W
        logdelta=log((diago/(delta*1.))+const)
    
    
        Xw=Xw.todense()
        Xw=np.array(Xw)
        Z1=np.multiply(Xw ,logdelta)
    
    
        Z1=np.array(Z1)
        Z1=Z1+log(pik+const)
    
        Z =np.zeros_like(Z1)
        Z[np.arange(len(Z1)), Z1.argmax(1)] = 1	
    
        Z=sp.lil_matrix(Z)
    
        del Z1
            
        ###mettre a jour proportion lignes et deltakk
        ############pik
    
        #n=Z.sum()
        pik=Z.sum(axis=0)
        pik=pik/nb_rows
    
        pik=np.squeeze(np.asarray(pik))
                
    
    
    
        ####deltakk
        
        Xw=m*W
    
        Xz=m.T*Z	
    
        
        Xw_k=Xw.sum(axis=0)
        Xz_k=Xz.sum(axis=0)
    
    
    
        den=  Xz_k.T * Xw_k
        diag_den=den.diagonal()
        sumden=diag_den.sum()
    
    
        den=1./(den+const)
        Num= ((Z.T * m) * W)
        deltakk=Num.multiply(den)
    
    
        diag_Num=Num.diagonal()
        sumpkk=np.sum(diag_Num)
    
    
    
        diago=deltakk.diagonal()
        deltakk=np.diag(np.squeeze(np.asarray(diago)))
    
        delta=(N-sumpkk)/(((1.*N*N)-sumden)+const)
        deltakk[deltakk == 0] = delta
    
        min_deltakk=np.min(deltakk[np.nonzero(deltakk)])
        deltakk[deltakk == 0] = min_deltakk * 0.01
    
            
        ##########cols
        Xz= m.T * Z
    
                
        logdelta=log((diago/(delta*1.))+const)
    
    
        Xz=Xz.todense()
        Xz=np.array(Xz)
        W1=np.multiply(Xz ,logdelta)
    
        W1=np.array(W1)
        W1=W1+log(pl+const)
    
        W =np.zeros_like(W1)
        W[np.arange(len(W1)), W1.argmax(1)] = 1	
    
        
    
        W=sp.lil_matrix(W)
        del W1
    
        ###############mettre a jour les prop colonnes et deltakk
        ############pl
    
        #d=W.sum()
    
        pl=W.sum(axis=0)
        pl=pl/nb_cols
    
        pl=np.squeeze(np.asarray(pl))
    
    
        ####deltakk
        
        Xw=m*W
    
        Xz=m.T*Z	
    
        
        Xw_k=Xw.sum(axis=0)
        Xz_k=Xz.sum(axis=0)
    
    
    
        den=  Xz_k.T * Xw_k
        diag_den=den.diagonal()
        sumden=diag_den.sum()
    
    
        den=1./(den+const)
        Num= ((Z.T * m) * W)
        deltakk=Num.multiply(den)
    
        deltakklbcem=deltakk
    
    
        diag_Num=Num.diagonal()
        sumpkk=np.sum(diag_Num)
    
    
    
        diago=deltakk.diagonal()
        deltakk=np.diag(np.squeeze(np.asarray(diago)))
    
        delta=(N-sumpkk)/(((1.*N*N)-sumden)+const)
        deltakk[deltakk == 0] = delta
    
        min_deltakk=np.min(deltakk[np.nonzero(deltakk)])
        deltakk[deltakk == 0] = min_deltakk * 0.01
    
            
        ###critere
    
        sumk=Z.sum(axis=0) * np.log(pik+const)[:,None]
        suml=W.sum(axis=0) * np.log(pl+const)[:,None]
        logdelta=log((diago/(delta*1.))+const)
                
        Lc= sp.lil_matrix(logdelta).multiply(sp.lil_matrix(Num.diagonal()))
    
        
        Lc=Lc.sum()+ N*(log(delta)-1) +  sumk.sum() + suml.sum()
    
        if len(news)>2 :
            if (Lc == news[len(news)-2]) :
                break	
    
        if np.abs(Lc - Linit)  > 1e-6 :
            print (Lc)
            Linit=Lc
            change=True
            news.append(Lc)
    
    
        em=em+1
    
    time_elapsed = (time.time() - time_start)
    
    
    ###############################evaluation measures#############################################
    #########################################nmi acc#########################################
    part=Z.todense().argmax(axis=1).tolist()
    part=[item for sublist in part for item in sublist]
    
    
    part2=W.todense().argmax(axis=1).tolist()
    part2=[item for sublist in part2 for item in sublist]
    
    
        
    n=normalized_mutual_info_score(labels, part)
    ari=adjusted_rand_score(labels, part)
    
    acc = accuracy(labels, part)
    print ("Accuracy ==> " + str(acc))
    print ("nmi ==> " + str(n))
    print ("adjusted rand index ==> " + str(ari))
    
    

    return pik,pl,deltakk, part,part2,news, acc, n, ari

################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################

def PLBcem(matrice, Z,W,labels ,K,L):
    m=matrice
    
    
    ####################################################################################nombre de co-clusters et autres petites choses a fixer##################################################
    m=sp.csr_matrix(m)
    
    
    
    nb_rows=m.shape[0]
    nb_cols=m.shape[1]
    
    
    
    
    const=1./(1.*m.sum()*m.sum())
    
    
    N=m.sum()
    
    
    ##################################################################################Initialisation#############################################################################################
    
    
    ##Z
    #Z=np.zeros((nb_rows,K))
    #Z_a=np.random.randint(K,size=nb_rows)
    #Z=np.zeros((nb_rows,K))
    #Z[np.arange(nb_rows) , Z_a]=1
    Z=sp.lil_matrix(Z)
    
    ###W
    #W=np.zeros((nb_cols,L))
    #W_a=np.random.randint(L,size=nb_cols)
    #W=np.zeros((nb_cols,L))
    #W[np.arange(nb_cols) , W_a]=1
    W=sp.lil_matrix(W)
    
    ############pik
    
    #n=Z.sum()
    
    pik=Z.sum(axis=0)
    pik=pik/nb_rows
    
    ############pl
    #d=W.sum()
    pl=W.sum(axis=0)
    pl=pl/nb_cols
    
    
    pik=np.squeeze(np.asarray(pik))
    pl=np.squeeze(np.asarray(pl))
    
    ##########gamakl
    
    Xw=m*W
    Xz=m.T*Z        
    Xw_l=Xw.sum(axis=0)
    print('Xw_l', Xw_l.shape)
    Xz_k=Xz.sum(axis=0)
    
    
    den=  Xz_k.T * Xw_l
    den=1/(den+const)
    gammakl=((Z.T * m) * W).multiply(den)
    print(gammakl)
    ##################################################################################co-clustering#############################################################################################
    
    change=True
    Linit=float(-inf)
    it=0
    news=[]
    gammakl = gammakl.todense()
    while change and it < 100:
        change=False
        
        ##################################################les lignes
    
        Xw=m * W
        #print('Xw',Xw.shape)
        
    
        ###step1
        logamma=np.log(gammakl.T+const)
        #print('logamma', logamma.shape)
        logamma=sp.lil_matrix(logamma)
        
        
        Z1=Xw * logamma
        Z1=Z1.toarray()
        Z1=Z1+np.log(pik+const)
        Z =np.zeros_like(Z1)
        Z[np.arange(len(Z1)), Z1.argmax(1)] = 1 
    
        Z=sp.lil_matrix(Z)
        
    
    
    
    
        ###step2
    
        #n=Z.sum()
        pik=Z.sum(axis=0)
        pik=pik/nb_rows
        pik=np.squeeze(np.asarray(pik))
    
    
        Xz=m.T * Z ####pour avoir les degres des clusters lignes
                
    
        Xw_l=Xw.sum(axis=0)
        Xz_k=Xz.sum(axis=0)
    
    
        
        Num=Z.T * Xw
        Den= Xz_k.T * Xw_l 
    
        
        Den=1/(Den+const)
        gammakl=Num.multiply(Den)
        gammakl= gammakl.todense()
        minval=np.min(gammakl[np.nonzero(gammakl)]) 
        gammakl[gammakl == 0] = minval*0.00000001
    
        
        
    
        ##################################################les colonnes
        Xz= m.T * Z 
        ###step1
    
        logamma=log(gammakl+const)
        logamma=sp.lil_matrix(logamma)
        
        W1= Xz * logamma
        W1=W1.toarray()
    
        W1=W1+np.log(pl+const)
    
        W=np.zeros_like(W1)
        W[np.arange(len(W1)), W1.argmax(1)] = 1 
        W=sp.lil_matrix(W)
        
    
    
        ###step2
    
        #d=W.sum()
        pl=W.sum(axis=0)
        pl=pl/nb_cols
        pl=np.squeeze(np.asarray(pl))
    
                
        Xw= m * W ####pour avoir les degres des clusters colonnes   
        Xw_l=Xw.sum(axis=0) 
        Xz_k=Xz.sum(axis=0)
    
    
        Num= W.T* Xz 
        Den=Xz_k.T  * Xw_l
    
        Den=1/(Den+const)   
    
        gammakl=(Num.T).multiply(Den)
        gammakl = gammakl.todense()
        minval=np.min(gammakl[np.nonzero(gammakl)])  #####remplacer les zeros dans gammakl par la valeur min dans gammakl*0.00000001
        gammakl[gammakl == 0] = minval*0.00000001
    
    
    
    
        ###Lc
        sumk=Z.sum(axis=0) * np.log(pik+const)[:,None]
        suml=W.sum(axis=0) * np.log(pl+const)[:,None]
        
        logamma=log(gammakl)
    
        Lcmat1=sp.lil_matrix(logamma).multiply(sp.lil_matrix(Num.T))
    
                
        Lc=Lcmat1.sum() + sumk.sum() + suml.sum()
                
    
    
    
        if np.abs(Lc - Linit)  > 1e-9 :
            news.append(Lc)
            print (Lc)
            Linit=Lc
            change=True
    
            
                
        it=it+1
        
    """"
    pik=pik[:,None]
    gammakl=np.array(gammakl)
    print gammakl
    print pik
    
    print (gammakl*pik).sum()
    
    print 1./m.sum()
    sys.exit()
    """
    print (Lc)
    
    ###############################evaluation measures#############################################
    #########################################nmi acc
    part=Z.todense().argmax(axis=1).tolist()
    part=[item for sublist in part for item in sublist]
    
    part2=W.todense().argmax(axis=1).tolist()
    part2=[item for sublist in part2 for item in sublist]
    
    
        
    n=normalized_mutual_info_score(labels, part)
    ari=adjusted_rand_score(labels, part)
    
    acc = accuracy(labels, part)
    print ("Accuracy ==> " + str(acc))
    print ("nmi ==> " + str(n))
    print ("adjusted rand index ==> " + str(ari))
    
    
    #################affiche la coube de convergence
    return pik,pl,gammakl, part,part2,news, acc, n, ari

