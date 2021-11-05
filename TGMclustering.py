# -*- coding: utf-8 -*-

"""
The module provides an implementation of TGM
a Tensor clustering algorithm by direct maximization of graph modularity.
"""
# License: BSD 3 clause

import numpy as np
from scipy import sparse
from sklearn.utils import check_random_state, check_array


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

    random_state = check_random_state(random_state)
    W_a = random_state.randint(n_clusters, size=n_cols)
    W = np.zeros((n_cols, n_clusters))
    W[np.arange(n_cols), W_a] = 1
    return W

class TGM():
    """ Tensor clustering by direct maximization of graph modularity.

    Parameters
    ----------
    n_clusters : int, optional, default: 2
        Number of co-clusters to form

    init : numpy array or scipy sparse matrix, \
        shape (n_features, n_clusters), optional, default: None
        Initial column labels

    max_iter : int, optional, default: 20
        Maximum number of iterations

    n_init : int, optional, default: 1
        Number of time the algorithm will be run with different
        initializations. The final results will be the best output of `n_init`
        consecutive runs in terms of modularity.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    tol : float, default: 1e-9
        Relative tolerance with regards to modularity to declare convergence

    Attributes
    ----------
    row_labels_ : array-like, shape (n_rows,)
        Bicluster label of each row


    modularity : float
        Final value of the modularity

    modularities : list
        Record of all computed modularity values for all iterations

    References
    ----------
    * Ailem M., Role F., Nadif M., Co-clustering Document-term Matrices by \
    Direct Maximization of Graph Modularity. CIKM 2015: 1807-1810
    """

    def __init__(self, n_clusters=2, init=None, max_iter=100, n_init=1,bool_fuzzy=False,fusion_method='Add',
                 tol=1e-9, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state
        self.bool_fuzzy = bool_fuzzy
        self.fusion_method = fusion_method
        self.row_labels_ = None
        self.modularity = -np.inf
        self.modularities = None
        self.paritionIteration = None



    def fit(self, X, y=None):
            """Perform co-clustering by direct maximization of graph modularity.

            Parameters
            ----------
            X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
                Matrix to be analyzed
            """

            random_state = check_random_state(self.random_state)

            # check_array(X, accept_sparse=True, dtype="numeric", order=None,
            #             copy=False, force_all_finite=True, ensure_2d=True,
            #             allow_nd=False, ensure_min_samples=self.n_clusters,
            #             ensure_min_features=self.n_clusters,
            #             warn_on_dtype=False, estimator=None)

            #if type(X) == np.ndarray:
            #    X = np.asarray(X)



            modularity = self.modularity
            row_labels_ = None

            seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
            for seed in seeds:
                self._fit_single(X, seed, y)
                if np.isnan(self.modularity):
                    raise ValueError("matrix may contain unexpected NaN values")
                # remember attributes corresponding to the best modularity
                if (self.modularity > modularity):
                    modularity = self.modularity
                    row_labels_ = self.row_labels_

            # update attributes
            self.modularity = modularity
            self.row_labels_ = row_labels_

            return self




    def _fit_single(self, X, random_state, y=None):
        """Perform one run of co-clustering by direct maximization of graph
        modularity.

        Parameters
        ----------
        X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
            Matrix to be analyzed
        """
        n = X[0].shape[0]
        if self.init is None:

            Z_init = random_init(self.n_clusters, n,random_state= random_state)

        else:
            Z_init = np.asarray(self.init, dtype=float)


        Z = np.zeros((n, self.n_clusters))


        v = len(X)
        list_indep= []
        list_N_v = []
        list_B  = []
        for v_ in range(v):
            X_ = X[v_]
            X_ = X_.astype(float)
            # Compute the modularity matrix
            row_sums = X_.sum(axis=1).reshape(n,1)
            col_sums = X_.sum(axis=0).reshape(1,n)

            N_v = X_.sum()
            list_N_v.append(N_v)

            indep_v = (row_sums.dot(col_sums)) / N_v
            print('indep_v', indep_v.shape)
            list_indep.append(indep_v)
            # B is a numpy matrix

            #B_v = (X_ - indep_v)
            B_v = (X_ - indep_v)* 1/X_.sum()
            list_B.append(B_v)


        self.modularities = []
        self.paritionIteration = []
        print("ok part 1")
        # Loop
        m_begin = float("-inf")
        change = True
        iteration = 0
        while change:
            print('Iteration N : ', iteration)
            change = False

            # Reassign rows
            if self.fusion_method == 'Add':
                B = np.zeros((n,n))
                for v_ in range(v):
                    B = B + list_B[v_]

            elif self.fusion_method == 'Pairwise':
                B = np.ones((n,n))
                for v_ in range(v):
                    B = B * list_B[v_]

            elif self.fusion_method == 'DotProduct':
                B = list_B[0]
                for v_ in range(1,v):
                    B = B.dot(list_B[v_])


            BZ = B.dot(Z_init)


            for idx in range(BZ.shape[0]):

                if self.bool_fuzzy == True:
                    # print("soft")
                    Z[idx, :] = BZ[idx, :] - np.amax(BZ[idx, :])
                    Z[idx, :] = np.exp(Z[idx, :]) / np.sum(np.exp(Z[idx, :]))
                else:
                    vectIndices = np.argmax(BZ, axis=1)
                    Z[idx, :] = 0
                    Z[idx, vectIndices[idx]] = 1


            ##########################
            # For link prediction
            n_times_n = (BZ).dot(Z.T)
            BZZ = []
            BZV =[]
            for v_ in range(v):
                BV = list_B[v_]
                nv_times_k = BV.dot(Z_init)
                BZV.append(nv_times_k)
                nv_times_n =(nv_times_k).dot(Z.T)
                BZZ.append(nv_times_n)

            ##########################
            k_times_k = (Z.T).dot(BZ)
            m_end = np.trace(k_times_k)
            iteration += 1
            if (np.abs(m_end - m_begin) > self.tol and
                    iteration < self.max_iter):
                self.modularities.append(m_end / np.sum(np.asarray(list_N_v)))
                self.paritionIteration.append(np.argmax(Z, axis=1).tolist())
                m_begin = m_end
                change = True
                Z_init = Z



        self.row_labels_ = np.argmax(Z, axis=1).tolist()
        self.bz = BZ
        self.BZV =BZV
        self.edgePred = n_times_n
        self.edgePredV = BZZ
        self.modularity = m_end / np.sum(np.asarray(list_N_v))
        self.nb_iterations = iteration