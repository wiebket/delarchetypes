#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:11:20 2018

@author: saintlyvi
"""

import pandas as pd
import feather
import os
from glob import glob
import sys

import delprocess.loadprofiles as lp

from .cluster import xBins
from ..support import data_dir, experiment_dir


def getExperiments(exp_root):
    """
    Retrieve experiments with root name exp_root from the results directory. 
    Returns list of unique experiments with root exp_root.
    """
    
    exps = glob(os.path.join(data_dir,'cluster_results',exp_root + '*.csv'))
    experiments = list(pd.Series([('_').join(x.split('/')[-1].split('_')[:-1]) for x in exps]).drop_duplicates(keep='last'))
    experiments.sort()
    
    return experiments


def getExpDetails(experiment_name):
    """
    """
    
    exp_root = '_'.join(experiment_name.split('_',2)[:2])
    prepro = experiment_name.split('_',2)[2]
    param_dir = os.path.join(experiment_dir, 'parameters')
    param_path = os.path.join(param_dir, exp_root + '.txt')
    header = open(param_path,'r')
    param = list()
    for line in header:
        if line.strip() != '':                # ignore blank lines
            param.append(eval(line))

    exp = pd.DataFrame(param[1:], columns=param[0])
    exp = exp.drop('range_n_clusters', axis=1)
    
    year_start = exp.loc[exp.preprocessing==prepro, 'start'].unique()[0]
    year_end = exp.loc[exp.preprocessing==prepro, 'end'].unique()[0]
    drop_0 = exp.loc[exp.preprocessing==prepro, 'drop_0'].unique()[0]
    bin_X = exp.loc[exp.preprocessing==prepro, 'bin_X'].unique()[0]
    
    return year_start, year_end, drop_0, prepro, bin_X, exp_root


def readResults():
    """
    """
    
    cluster_results = pd.read_csv('results/cluster_results.csv')
    cluster_results.drop_duplicates(subset=[
            'dbi','mia','experiment_name','elec_bin'],keep='last',inplace=True)
    cluster_results = cluster_results[cluster_results.experiment_name != 'test']
    cluster_results['score'] = (
            cluster_results.dbi * cluster_results.mia / cluster_results.silhouette)
    cluster_results['clusters'] = cluster_results.loc[:, 'n_clust'].where(
            cluster_results['n_clust'] > 0,
            cluster_results['som_dim']**2)
    
    return cluster_results


def weightScore(cluster, score):
    """ 
    ***adpated from http://pbpython.com/weighted-average.html***
    http://stackoverflow.com/questions/10951341/pandas-dataframe-aggregate-function-using-multiple-columns
    In rare instance, we may not have weights, so just return the mean. Customize this if your business case
    should return otherwise.
    """
    d = cluster[score]
    w = cluster['total_sample']
    try:
        return (d * w).sum() / w.sum()
        
    except ZeroDivisionError:
    
        return d.mean()


def selectClusters(cluster_results, n_best, experiment='all' ):
    """
    """
    
    if experiment=='all':
        exc = cluster_results.loc[cluster_results.score > 0,:]
    else:
        exc = cluster_results.loc[(cluster_results.experiment_name == experiment) &
                                  (cluster_results.score>0), :]

    experiment_clusters = pd.DataFrame()
    
    for e in exc.experiment_name.unique():    
        if int(e[3]) < 4:    
            i_ec = exc.loc[exc.experiment_name == e].groupby(['experiment_name', 'som_dim', 'n_clust'
                          ]).mean().reset_index() 
            experiment_clusters = experiment_clusters.append(i_ec, sort=True)
            
        elif int(e[3]) >= 4:
            temp_ec = exc.loc[exc.loc[exc.experiment_name == e].groupby(['experiment_name', 'som_dim', 
                              'elec_bin'])['score'].idxmin(), ['experiment_name', 'som_dim', 'n_clust', 'elec_bin', 'dbi', 'mia', 'silhouette', 'score', 'total_sample']]
            
            i_ec = temp_ec.groupby(['experiment_name', 'som_dim'])['n_clust'].mean().reset_index(name='n_clust')
            for s in ['dbi', 'mia', 'silhouette', 'score']:
                i_ec.loc[:,s] = weightScore(temp_ec, s)
            
#            i_ec = temp_ec.groupby(['experiment_name', 'som_dim']).mean().drop(columns ='total_sample').reset_index()
            experiment_clusters = experiment_clusters.append(i_ec, sort=True) 
        
    best_clusters = experiment_clusters.nsmallest(columns='score',n=n_best).reset_index(drop=True).reindex(
                        ['som_dim','n_clust','dbi','mia','silhouette','score','experiment_name'],axis=1)

    best_clusters.insert(0, 'experiment', best_clusters['experiment_name'].apply(lambda x: x.split('_', 1)[0][3]))
    best_clusters.insert(1, 'algorithm', best_clusters['experiment_name'].apply(lambda x: x.split('_', 2)[1]))
    prepro = best_clusters['experiment_name'].apply(lambda x: x.split('_', 2)[2] if x.count('_')>1 else None)
    best_clusters.insert(2, 'pre_processing', prepro)
    
    return best_clusters


def exploreAMDBins(experiment, elec_bin=None):
    """
    """
    
    cluster_results = readResults()
    if elec_bin is None:
        exc = cluster_results[['experiment_name','som_dim','n_clust','elec_bin',
                               'dbi','mia','silhouette','score','total_sample']]
    else:
        exc = cluster_results.loc[cluster_results['elec_bin']==elec_bin,['experiment_name','som_dim','n_clust','elec_bin','dbi','mia','silhouette','score','total_sample']]
    
    temp_ec = exc.loc[exc.loc[exc.experiment_name.str.contains(experiment)].groupby(['experiment_name', 'som_dim','elec_bin'])['score'].idxmin(), ['experiment_name','som_dim','n_clust','elec_bin','dbi','mia', 'silhouette','score','total_sample']]
    ordered_cats = [i for i in exc.elec_bin.unique() if i in temp_ec.elec_bin.unique()]
    temp_ec.elec_bin = temp_ec.elec_bin.astype('category')
    temp_ec.elec_bin = temp_ec.elec_bin.cat.reorder_categories(ordered_cats, ordered=True)
    temp_ec.sort_values('elec_bin', inplace=True)
    
    temp_ec.set_index(['experiment_name','som_dim','n_clust'],inplace=True)
    ec_amd = temp_ec.loc[:,['elec_bin','score','total_sample']]
   
    return ec_amd


def getLabels(experiment, n_best=1):
    
    year_start, year_end, drop, prepro, bin_X, exp_root = getExpDetails(experiment)
    
    label_dir = os.path.join(data_dir, 'cluster_evaluation', 'best_labels')
    os.makedirs(label_dir, exist_ok=True)

    if drop == False:
        label_path = os.path.join(label_dir, experiment+'BEST'+str(n_best)+'_labels.feather')
    elif drop == True:
        label_path = os.path.join(label_dir, experiment+'drop0BEST'+str(n_best)+'_labels.feather')

    if os.path.exists(label_path) is True:
        XL = feather.read_dataframe(label_path).set_index(['ProfileID','date'])
    
    else:    
        X = lp.genX([1994,2014], drop_0=drop)
        print('Creating labelled dataframe...')
        
        if int(experiment[3]) < 4:
            path = glob(os.path.join(data_dir, 'cluster_results', experiment+'_*_labels.feather'))[0]
            labels = feather.read_dataframe(path).iloc[:, n_best-1]
            X.reset_index(inplace=True)
            X['k'] = labels + 1
            X['elec_bin'] = 'all'
            XL = X
    
        elif int(experiment[3]) >= 4: #reconstruct full X for experiment 4, 5
            Xbin = xBins(X, bin_X)
            XL = pd.DataFrame()
    
            for b, ids in Xbin.items():
                paths = glob(os.path.join(data_dir, 'cluster_results', experiment+'*'+b+'_labels.feather'))
                paths.sort()
                path = paths[0]
                labels = feather.read_dataframe(path).iloc[:, n_best-1]
                
                if XL.empty == True:
                    cluster_add = 1
                else:
                    cluster_add = XL['k'].max() + 1
                A = X.loc[ids,:].reset_index()   
                A['k'] = labels + cluster_add
                A['elec_bin'] = b
                XL = XL.append(A)
            
            del Xbin
                
        feather.write_dataframe(XL, label_path)
        XL.set_index(['ProfileID','date'], inplace=True)          

        del X
    
    return XL.sort_index()


def realCentroids(experiment, xlabel=None, n_best=1):
    """
    """
    
    year_start, year_end, drop_0, prepro, bin_X, exp_root = getExpDetails(experiment)

    os.makedirs(os.path.join(data_dir, 'cluster_evaluation', 'best_centroids'), exist_ok = True)
    centpath = os.path.join(data_dir, 'cluster_evaluation', 'best_centroids', 
                            experiment+'BEST'+str(n_best)+'_centroids.csv')
    try:
        centroids  = pd.read_csv(centpath, index_col='k')
    
    except:
        centroids = xlabel.groupby('k').mean()
        centroids['elec_bin'] = [xlabel.loc[xlabel.k==i,'elec_bin'].iloc[0] for i in centroids.index]
        centroids['cluster_size'] = xlabel.groupby('k')['0'].count()
        centroids['experiment'] = experiment
        centroids['n_best'] = n_best
        
        ordered_cats = centroids.elec_bin.unique()
        centroids.elec_bin = centroids.elec_bin.astype('category')
        centroids.elec_bin = centroids.elec_bin.cat.reorder_categories(ordered_cats, ordered=True)
    
        centroids.to_csv(centpath, index=True)
        print('Real centroids computed and recorded.')
    
    return centroids


def rebinCentroids(centroids):
    """
    """
    
    try:
        centroids['AMD'] = centroids.iloc[:,0:24].sum(axis=1)

        monthly_consumption_bins = [1, 50, 150, 400, 600, 1200, 2500, 4000]
        daily_demand_bins = [x /30*1000/230 for x in monthly_consumption_bins]
        bin_labels = ['{0:.0f}-{1:.0f} mean_amd'.format(x,y) for x, y in zip(monthly_consumption_bins[:-1], monthly_consumption_bins[1:])]
        centroids['elec_bin'] = pd.cut(centroids['AMD'], daily_demand_bins, labels=bin_labels, right=False)
        centroids['elec_bin'] = centroids.elec_bin.where(~centroids.elec_bin.isna(), '0-1')
        centroids['elec_bin'] = centroids.elec_bin.astype('category')
        ordered_cats = ['0-1']+[i for i in bin_labels if i in centroids.elec_bin.cat.categories]
        centroids.elec_bin.cat.reorder_categories(ordered_cats, ordered=True,inplace=True)
    except:
        pass

    return centroids


def mapBins(centroids):
    """
    """
    
    centroids['dd'] = centroids.iloc[:,0:24].sum(axis=1)
    new_cats = centroids.groupby('elec_bin')['dd'].mean().reset_index()
    new_cats['bin_labels'] = pd.Series(['{0:.0f} A mean_dd'.format(x) for x in new_cats['dd']])
    sorted_cats = new_cats.sort_values('dd')    
    mapper = dict(zip(sorted_cats['elec_bin'], sorted_cats['bin_labels']))
    
    return mapper


def renameBins(data, centroids):
    """
    """
    
    mapper = mapBins(centroids)
    data['elec_bin'] = data['elec_bin'].apply(lambda x:mapper[x])
    data['elec_bin'] = data['elec_bin'].astype('category')
    data.elec_bin.cat.reorder_categories(mapper.values(), ordered=True,inplace=True)
    data.index.name = 'k'
    data.sort_values(['elec_bin','k'], inplace=True)    
    
    return data   


def clusterColNames(data):    
    """
    """
    
    data.columns = ['Cluster '+str(x) for x in data.columns]
    
    return data