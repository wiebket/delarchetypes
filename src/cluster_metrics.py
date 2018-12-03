#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 10:53:07 2018

@author: saintlyvi
"""

import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances#, pairwise_distances_chunked

from sklearn.utils import check_X_y, safe_indexing
from sklearn.preprocessing import LabelEncoder

#Average Squared-Loss Mutual Information Error (SMI),
#Violation rate of Root Squared Error (VRSE)
#Modified Dunn Index (MDI) 
#Cluster Dispersion Indicator (CDI)

def check_number_of_labels(n_labels, n_samples):
    if not 1 < n_labels < n_samples:
        raise ValueError("Number of labels is %d. Valid values are 2 "
                         "to n_samples - 1 (inclusive)" % n_labels)


def davies_bouldin_score(X, labels):
    """ FROM SKLEARN PRE-RELEASE
    Computes the Davies-Bouldin score.
    The score is defined as the ratio of within-cluster distances to
    between-cluster distances.
    Read more in the :ref:`User Guide <davies-bouldin_index>`.
    Parameters
    ----------
    X : array-like, shape (``n_samples``, ``n_features``)
        List of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like, shape (``n_samples``,)
        Predicted labels for each sample.
    Returns
    -------
    score: float
        The resulting Davies-Bouldin score.
    References
    ----------
    .. [1] `Davies, David L.; Bouldin, Donald W. (1979).
       "A Cluster Separation Measure". IEEE Transactions on
       Pattern Analysis and Machine Intelligence. PAMI-1 (2): 224-227`_
    """
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, _ = X.shape
    n_labels = len(le.classes_)
    check_number_of_labels(n_labels, n_samples)

    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(X[0])), dtype=np.float)
    for k in range(n_labels):
        cluster_k = safe_indexing(X, labels == k)
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        intra_dists[k] = np.average(pairwise_distances(
            cluster_k, [centroid]))

    centroid_distances = pairwise_distances(centroids)

    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0

    score = (intra_dists[:, None] + intra_dists) / centroid_distances
    score[score == np.inf] = np.nan
    return np.mean(np.nanmax(score, axis=1))

def mean_index_adequacy(X, labels):
    """
    based on methods described in Optimal Selection of Clustering Algorithm via Multi-Criteria Decision 
    Analysis (MCDA) for Load Profiling Applications DOI: 10.3390/app8020237
    
    """
   
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, _ = X.shape
    n_labels = len(le.classes_)
    check_number_of_labels(n_labels, n_samples)

    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(X[0])), dtype=np.float)
    for k in range(n_labels):
        cluster_k = safe_indexing(X, labels == k)
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        # calculate cluster dispersion using RMS
        intra_dists[k] = np.sqrt(np.mean(pairwise_distances(cluster_k, [centroid])**2))

    # get mean of distance values    
    mia = np.sqrt(np.mean(intra_dists**2))
    
    return mia

def cluster_dispersion_index(X, labels):
    """
    based on methods described in 10.1109/TPWRS.2006.873122
    
    """
    #NB MUST CHANGE THIS TO PAIRWISE_DISTANCES_CHUNKED TO WORK. Not yet released.
    #Don't use this function to evaluate large datasets!
    n_clusters = len(np.bincount(labels))
    cluster_k = [X[labels == k] for k in range(n_clusters)]
    centroids = [np.mean(k, axis = 0) for k in cluster_k]
    
    Dclust = np.sqrt(np.sum(
                            [np.sqrt(np.sum(
                                    np.square(euclidean_distances(cluster_k[i]) ) )/(2*len(cluster_k[i]) )
                                    ) for i in range(n_clusters)]) / n_clusters )
    Dcent = np.sqrt(np.sum(np.square(euclidean_distances(centroids) ) )/(2*n_clusters) )
    
    cdi = Dclust/Dcent
    
    return cdi

