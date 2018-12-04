#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday 3 December 2018

@author: Wiebke Toussaint
"""

import os
import pandas as pd
import numpy as np
import feather
import time
from datetime import date

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import somoclu

from dlrprocessing.loadprofiles import resampleProfiles

from metrics import mean_index_adequacy, davies_bouldin_score
from .support import cluster_dir, results_dir

def progress(n, stats):
    """Report progress information, return a string."""
    s = "%s : " % (n)                    
    s += "\nsilhouette: %(silhouette).3f " % stats
    s += "\ndbi: %(dbi).3f " % stats
    s += "\nmia: %(mia).3f " % stats  
    return print(s)
    
def clusterStats(cluster_stats, n, X, cluster_labels, preprocessing, transform, tic, toc):   
   
    stats = {'n_sample': 0,
         'cluster_size': [],
         'silhouette': 0.0,
         'dbi': 0.0,
         'mia': 0.0,
         'all_scores': 0.0,
#             'cdi': 0.0,
         't0': time.time(),
         'batch_fit_time': 0.0,
         'total_sample': 0}

    cluster_stats[n] = stats
    try:
        cluster_stats[n]['total_sample'] += X.shape[0]
        cluster_stats[n]['n_sample'] = X.shape[0]
        cluster_stats[n]['silhouette'] = silhouette_score(X, cluster_labels, sample_size=10000)
        cluster_stats[n]['dbi'] = davies_bouldin_score(X, cluster_labels)
        cluster_stats[n]['mia'] = mean_index_adequacy(X, cluster_labels)
        #cluster_stats[n_clusters][y]['cdi'] =cluster_dispersion_index(Xbatch, cluster_labels) DON'T RUN LOCALLY!! - need to change to chunked alogrithm once released
        cluster_stats[n]['cluster_size'] = np.bincount(cluster_labels)
        cluster_stats[n]['batch_fit_time'] = toc - tic
        cluster_stats[n]['preprocessing'] = preprocessing
        cluster_stats[n]['transform'] = transform
        cluster_stats[n]['all_scores'] = cluster_stats[n]['dbi']*cluster_stats[n]['mia']/cluster_stats[n]['silhouette']

        s = "%s : " % (n)                    
        s += "\nsilhouette: %(silhouette).3f " % stats
        s += "\ndbi: %(dbi).3f " % stats
        s += "\nmia: %(mia).3f " % stats
        print(s)

    except:
        print('Could not compute clustering stats for n = ' + str(n))
        pass

    return cluster_stats

def saveResults(experiment_name, cluster_stats, cluster_centroids, som_dim, elec_bin, save=True):
    """
    Saves cluster stats results and centroids for a single clustering iteration. 
    Called inside kmeans() and som() functions.
    """

    for k, v in cluster_stats.items():
        n = k
                        
    evals = pd.DataFrame(cluster_stats).T
    evals['experiment_name'] = experiment_name
    evals['som_dim'] = som_dim
    evals['n_clust'] = n
    evals['elec_bin'] = elec_bin
    eval_results = evals.drop(labels='cluster_size', axis=1).reset_index(drop=True)
#    eval_results.rename({'index':'k'}, axis=1, inplace=True)
    eval_results[['dbi','mia','silhouette']] = eval_results[['dbi','mia','silhouette']].astype(float)
    eval_results['date'] = date.today().isoformat()
#    eval_results['best_clusters'] = None

    centroid_results = pd.DataFrame(cluster_centroids)   
    centroid_results['experiment_name'] = experiment_name
    centroid_results['som_dim'] = som_dim
    centroid_results['n_clust'] = n
    centroid_results['elec_bin'] = elec_bin
    try:
        centroid_results['cluster_size'] = evals['cluster_size'][n]
    except:
        centroid_results['cluster_size'] = np.nan
    centroid_results.reset_index(inplace=True)
    centroid_results.rename({'index':'k'}, axis=1, inplace=True)
    centroid_results['date'] = date.today().isoformat()
    
    #3 Save Results
    if save is True:
        os.makedirs(results_dir, exist_ok=True)    
        erpath = os.path.join(results_dir, 'cluster_results.csv')    
        if os.path.isfile(erpath):
            eval_results.to_csv(erpath, mode='a', index=False, header=False)
        else:
            eval_results.to_csv(erpath, index=False)

        os.makedirs(cluster_dir, exist_ok=True)   
        crpath = os.path.join(cluster_dir, experiment_name + '_centroids.csv')    
        if os.path.isfile(crpath):
            centroid_results.to_csv(crpath, mode='a', index=False, header=False)
        else:
            centroid_results.to_csv(crpath, index=False)
        
        print('Results saved for', experiment_name, str(som_dim), str(n))
    
    return eval_results, centroid_results

def xBins(X, bin_type):

    if bin_type == 'amd':
        Xdd_A = X.sum(axis=1)
        Xdd = Xdd_A*230/1000
        XmonthlyPower = resampleProfiles(Xdd, interval='M', aggfunc='sum')
        Xamd = resampleProfiles(XmonthlyPower, interval='A', aggfunc='mean').reset_index().groupby('ProfileID').mean()
        Xamd.columns=['amd']
        
        amd_bins = [0, 1, 50, 150, 400, 600, 1200, 2500, 4000]    
        bin_labels = ['{0:.0f}-{1:.0f}'.format(x,y) for x, y in zip(amd_bins[:-1], amd_bins[1:])]    
        Xamd['bins'] = pd.cut(Xamd.amd, amd_bins, labels=bin_labels, right=True, include_lowest=True)
        
        Xbin_dict = dict()
        for c in Xamd.bins.cat.categories:
            Xbin_dict[c] = Xamd[Xamd.bins==c].index.values
        
        del Xdd_A, Xdd, XmonthlyPower, Xamd
        
    if bin_type == 'integral':
        Xint = normalize(X).cumsum(axis=1)
        Xintn = pd.DataFrame(Xint, index=X.index)
        Xintn['max'] = X.max(axis=1)
        clusterer = MiniBatchKMeans(n_clusters=8, random_state=10)
        clusterer.fit(np.array(Xintn))
        cluster_labels = clusterer.predict(np.array(Xintn))
        labl = pd.DataFrame(cluster_labels, index=X.index) 
        Xbin_dict = dict()
        for c in labl[0].unique():
            Xbin_dict['bin'+str(c)] = labl[labl[0]==c].index.values
            
    return Xbin_dict

def preprocessX(X, norm=None):  
    
    if norm == 'unit_norm': #Kwac et al 2013
        Xnorm = normalize(X)
    elif norm == 'zero-one': #Dent et al 2014
        Xnorm = np.array(X.divide(X.max(axis=1), axis=0))
    elif norm == 'demin': #Jin et al 2016
        Xnorm = normalize(X.subtract(X.min(axis=1), axis=0))
    elif norm == 'sa_norm': #Dekenah 2014
        Xnorm = np.array(X.divide(X.mean(axis=1), axis=0))
    else:
        Xnorm = np.array(X)
    
    #Xnorm.fillna(0, inplace=True)
    Xnorm[np.isnan(Xnorm)] = 0
        
    return Xnorm

def kmeans(X, range_n_clusters, top_lbls=10, preprocessing = None, bin_X=False, experiment_name=None):
    """
    This function applies the MiniBatchKmeans algorithm from sklearn on inputs X for range_n_clusters.
    If preprossing = True, X is normalised with sklearn.preprocessing.normalize()
    Returns cluster stats, cluster centroids and cluster labels.
    """

    if experiment_name is None:
        save = False
    else:
        if preprocessing is None:
            pass
        else:
            experiment_name = experiment_name+'_'+ preprocessing
        save = True
    
    #apply pre-binning
    if bin_X != False:
        Xbin = xBins(X, bin_X)
    else:
        Xbin = {'all':X}

    for b, ids in Xbin.items():
        try:
            A = X.loc[ids,:]
        except:
            A = ids

        #apply preprocessing    
        A = preprocessX(A, norm=preprocessing)

        centroids = pd.DataFrame()
        stats = pd.DataFrame() 
        cluster_lbls = pd.DataFrame()

        dim = 0 #set dim to 0 to match SOM formating  
        cluster_lbls_dim = {}
        stats_dim = pd.DataFrame()
        
        for n_clust in range_n_clusters:
            
            clusterer = MiniBatchKMeans(n_clusters=n_clust, random_state=10)
                        
            #train clustering algorithm
            tic = time.time()        
            clusterer.fit(A)
            cluster_labels = clusterer.predict(A)
            toc = time.time()
            
             ## Calculate scores
            cluster_stats = clusterStats({}, n_clust, A, cluster_labels, 
                                         preprocessing = preprocessing, transform = None,
                                         tic = tic, toc = toc)        
            cluster_centroids = clusterer.cluster_centers_ 
            
            eval_results, centroid_results = saveResults(experiment_name, cluster_stats,
                                                          cluster_centroids, dim, b, save)
            
            stats_dim = stats_dim.append(eval_results)
            centroids = centroids.append(centroid_results)
    
            cluster_lbls_dim[n_clust] = cluster_labels
    
        #outside n_clust loop
        best_clusters, best_stats = bestClusters(cluster_lbls_dim, stats_dim, top_lbls)
        cluster_lbls = pd.concat([cluster_lbls, best_clusters], axis=1)
        stats = pd.concat([stats, best_stats], axis=0)
        
        stats.reset_index(drop=True, inplace=True)

        if save is True:
            saveLabels(cluster_lbls, stats)
    
    return stats, centroids, cluster_lbls        

def som(X, range_n_dim, top_lbls=10, preprocessing = None, bin_X=False, transform=None, experiment_name=None, **kwargs):
    """
    This function applies the self organising maps algorithm from somoclu on inputs X over square maps of range_n_dim.
    If preprossing = True, X is normalised with sklearn.preprocessing.normalize()
    If kmeans = True, the KMeans algorithm from sklearn is applied to the SOM and returns clusters
    kwargs can be n_clusters = range(start, end, interval) OR list()
    Returns cluster stats, cluster centroids and cluster labels.
    """

    for dim in range_n_dim: 
        limit = int(np.sqrt(len(X)/20))
        if dim > limit: #verify that number of nodes are sensible for size of input data
            return print('Input size too small for map. Largest n should be ' + str(limit))
        else:
            pass
        
    if experiment_name is None:
        save = False
    else:
        if preprocessing is None:
            pass
        else:
            experiment_name = experiment_name+'_'+ preprocessing
        save = True

    #apply pre-binning
    if bin_X != False:
        Xbin = xBins(X, bin_X)
    else:
        Xbin = {'0-4000':X}

    for b, ids in Xbin.items():
        try:
            A = X.loc[ids,:]
        except:
            A = ids
        #apply preprocessing    
        A = preprocessX(A, norm=preprocessing)

        centroids = pd.DataFrame()
        stats = pd.DataFrame() 
        cluster_lbls = pd.DataFrame()

        for dim in range_n_dim: 
            
            cluster_lbls_dim = {}
            stats_dim = pd.DataFrame()        
            nrow = ncol = dim
            tic = time.time()
    
            #train clustering algorithm
            som = somoclu.Somoclu(nrow, ncol, compactsupport=False, maptype='planar')
            som.train(A)
            toc = time.time()
    
            if transform == None:
                n_clust = [0]    
            elif transform == 'kmeans':
                if kwargs is None:
                    n_clust = [10]
                else:
                    for key, value in kwargs.items(): #create list with number of clusters for kmeans
                        if key == 'n_clusters':
                            n_clust = value
            else:
                return('Cannot process this transform algorithm')
                
            for n in n_clust:
                if n == 0:
                    #create empty matrix the size of the SOM
                    m = np.arange(0, nrow*ncol, 1).reshape(nrow, ncol) 
                else:
                    clusterer = KMeans(n_clusters=n, random_state=10)
                    som.cluster(algorithm=clusterer)
                    m = som.clusters
                #get cluster of SOM node and assign to input vecors based on bmus
                k = [m[som.bmus[i][1],som.bmus[i][0]] for i in range(0, len(som.bmus))] 
                c = pd.DataFrame(A).assign(cluster=k).groupby('cluster').mean()
                    
                #calculate scores
                cluster_stats = clusterStats({}, n, A, cluster_labels = k, preprocessing = preprocessing,
                                             transform = transform, tic = tic, toc = toc)
                cluster_centroids = np.array(c)
                
                eval_results, centroid_results = saveResults(experiment_name, cluster_stats,
                                                             cluster_centroids, dim, b, save)
    
                stats_dim = stats_dim.append(eval_results)
                centroids = centroids.append(centroid_results)
    
                cluster_lbls_dim[n] = k
            
            #outside n_clust loop
            best_clusters, best_stats = bestClusters(cluster_lbls_dim, stats_dim, top_lbls)
            cluster_lbls = pd.concat([cluster_lbls, best_clusters],axis=1)
            stats = pd.concat([stats, best_stats], axis=0)
            
        stats.reset_index(drop=True, inplace=True)
        if save is True:
            saveLabels(cluster_lbls, stats)
    
    return stats, centroids, cluster_lbls

def bestClusters(cluster_lbls, stats, top_lbls):

    labels = pd.DataFrame(cluster_lbls)
    
    if len(labels) > top_lbls:    
#        best_lbls = stats.nsmallest(columns=['dbi','mia'], n=top_lbls).nlargest(columns='silhouette',
#                                          n=top_lbls)[['n_clust','som_dim']].reset_index(drop=True)
#        b = stats.dbi*stats.mia/stats.silhouette
        stats.all_scores = stats.all_scores.astype('float')
        best_lbls = stats[stats.all_scores>0].nsmallest(columns='all_scores', n=top_lbls 
                         ).reset_index(drop=True)
        best_clusters = labels.loc[:, best_lbls['n_clust'].values]    
    
    else:
        best_lbls = stats[['n_clust','som_dim','elec_bin']]
        best_clusters = labels
    
#    best_clusters.columns = pd.MultiIndex.from_arrays([best_lbls['som_dim'], best_lbls['n_clust']],names=('som_dim','n_clust'))    
    stats.loc[stats['n_clust'].isin(best_lbls['n_clust'].values), 'best_clusters'] = 1
    
    return best_clusters, stats
       
def saveLabels(cluster_lbls, stats):    

    experiment_name = stats.experiment_name[0]
    elec_bin = stats.elec_bin[0]
    best_lbls = stats.loc[stats.best_clusters==1,['n_clust','som_dim','elec_bin']]
    best_lbls['experiment_name'] = experiment_name     
      
#    cluster_lbls[['ProfileID','date']] = pd.DataFrame(X).reset_index()[['ProfileID','date']]
#    cluster_lbls.set_index(['ProfileID','date'], inplace=True)
#    cluster_lbls.columns = pd.MultiIndex.from_arrays([best_lbls['som_dim'], best_lbls['n_clust']],names=('som_dim','n_clust'))
#    cluster_lbls.dropna(inplace=True)    
    cols = []
# TO DO this column order is wrong!!
    for i, j in zip(best_lbls['som_dim'],best_lbls['n_clust']):
        cols.append(str(i)+'_'+str(j))
    print(cols)
    cluster_lbls.columns = cols

    wpath = os.path.join(cluster_dir, experiment_name + '_' + elec_bin + '_labels.feather')
    feather.write_dataframe(cluster_lbls, wpath)
    
    blpath = os.path.join(results_dir, 'best_clusters.csv')
    if os.path.isfile(blpath):
        best_lbls.to_csv(blpath, mode='a', index=False, header=False)
    else:
        best_lbls.to_csv(blpath, index=False)
    
    return print('Labels for best '+experiment_name+' clusters saved')
