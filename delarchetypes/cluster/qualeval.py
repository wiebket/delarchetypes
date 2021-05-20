#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 08:37:54 2018

@author: saintlyvi
"""

import pandas as pd
import numpy as np
import datetime as dt
from math import log
import os
import peakutils

import delprocess.surveys as surveys

from ..support import data_dir

def consumptionError(xlabel, centroids, compare='total'):
    """
    Calculate error metrics for total daily consumption (compare=total) or peak daily consumption (compare=peak).
    Returns 
    mean absolute percentage error, 
    median absolute percentage error, 
    median log accuracy ratio (Q=predicted/actual)
    median symmetric accuracy
    """
    
    cent = centroids.iloc[:,0:24]
    
    if compare == 'total':
        X_dd = pd.concat([xlabel.iloc[:,list(range(0,24))].sum(axis=1), xlabel.iloc[:,-2]], axis=1, keys=['DD','k'])
        cent_dd = cent.sum(axis=1).rename_axis('k',0).reset_index(name='DD')
    elif compare == 'peak':
        X_dd = pd.concat([xlabel.iloc[:,list(range(0,24))].max(axis=1), xlabel.iloc[:,-2]], axis=1, keys=['DD','k'])
        cent_dd = cent.max(axis=1).rename_axis('k',0).reset_index(name='DD')

    X_dd['ae'] = 0
    X_dd['logq'] = 0
    for y in cent_dd.itertuples(): 
        X_dd.loc[X_dd.k==y[1],'ae'] = [abs(x-y[2]) for x in X_dd.loc[X_dd.k==y[1],'DD']]
        try:
            X_dd.loc[X_dd.k==y[1],'logq'] = [log(y[2]/x) for x in X_dd.loc[X_dd.k==y[1],'DD']]
        except:
            print('Zero values. Could not compute log(Q) for cluster', str(y[1]))
            X_dd.loc[X_dd.k==y[1],'logq'] = np.inf

    X_dd['ape'] = X_dd.ae/X_dd.DD
    X_dd['alogq'] = X_dd['logq'].map(lambda x: abs(x))
            
    mape = X_dd.groupby('k')['ape'].mean()*100
    mdape = X_dd.groupby('k')['ape'].agg(np.median)*100
    mdlq = X_dd.groupby('k')['logq'].agg(np.median)
    mdsyma = np.expm1(X_dd.groupby('k')['alogq'].agg(np.median))*100
    
    del X_dd

    #create data to write to file
    write_eval = pd.DataFrame([mape, mdape, mdlq, mdsyma], index=['mape', 'mdape', 'mdlq', 'mdsyma']).T
    write_eval['compare'] = compare
    write_eval['experiment'] = centroids['experiment'].unique()[0]
    write_eval['n_best'] = centroids['n_best'].unique()[0]
    
    cepath = os.path.join(data_dir, 'cluster_evaluation', 'consumption_error.csv')
    if os.path.isfile(cepath):
        write_eval.to_csv(cepath, mode='a', index=True, header=False)
    else:
        write_eval.to_csv(cepath, index=True)
    print('Consumption error output recorded.')
           
    return #mape, mdape, mdlq, mdsyma


def centroidPeaks(centroids):
    """
    """
    
    cents = centroids.iloc[:, 0:24]
    cent_peak = dict()
    for i in cents.iterrows():
        h = peakutils.peak.indexes(i[1], thres=0.5, min_dist=1)
        val = cents.iloc[i[0]-1, h].values
        cent_peak[i[0]] = dict(zip(h,val))
        
    return cent_peak


def peakCoincidence(xlabel, centroids):
    """
    """
    
    mod_xl = xlabel.drop(columns='elec_bin')
    
    try:
        #get peakcoincidence from csv
        data=pd.read_csv(os.path.join(data_dir, 'cluster_evaluation', 'peak_coincidence.csv'))
        peak_eval = data.loc[(data['experiment']==centroids['experiment'].unique()[0])& 
                             (data['n_best']==centroids['n_best'].unique()[0]), :]
        peak_eval = peak_eval.drop_duplicates(subset=['k', 'experiment','n_best'], 
                                              inplace=False, keep='last')
        if len(peak_eval) == 0:
            raise Exception
    except:
        X2 = pd.concat([mod_xl.iloc[:,list(range(0,24))], mod_xl.iloc[:,-1]], axis=1)
        X2.columns = list(range(0,24))+['k']
        
        cent_peak = centroidPeaks(centroids)
    
        clusters = X2.iloc[:,-1].unique()
        clusters.sort()
        X_peak = dict()
        for c in clusters:
            X_k = X2.loc[X2.k == c]      
            X_k.drop(columns='k', inplace=True)
            peak_count = 0
            for i in X_k.iterrows():
                h = peakutils.peak.indexes(i[1], thres=0.5, min_dist=1)
                peak_count += len(set(cent_peak[c]).intersection(set(h)))
            X_peak[c] = peak_count / len(X_k)
            print('Mean peak coincidence computed for cluster',str(c))
    
        peak_eval = pd.DataFrame(list(X_peak.items()), columns=['k','mean_coincidence'])
        count_cent_peaks = [len(cent_peak[i].keys()) for i in cent_peak.keys()]
        peak_eval['coincidence_ratio'] = peak_eval.mean_coincidence/count_cent_peaks #normalise for number of peaks
        peak_eval['experiment'] = centroids['experiment'].unique()[0]
        peak_eval['n_best'] = centroids['n_best'].unique()[0]
        
        pcpath = os.path.join(data_dir, 'cluster_evaluation', 'peak_coincidence.csv')
        if os.path.isfile(pcpath):
            peak_eval.to_csv(pcpath, mode='a', index=False, header=False)
        else:
            peak_eval.to_csv(pcpath, index=False)
        
        del X2    
    
    return peak_eval


def meanError(metric_vals):  
    """
    """
    
    err = metric_vals.where(~np.isinf(metric_vals)).mean()    
    return err


def demandCorr(xlabel, compare='total'):
    """
    """
	
    mod_xl = xlabel.drop(columns='elec_bin')
    
    if compare == 'total':
        data = pd.concat([mod_xl.iloc[:,list(range(0,24))].sum(axis=1), mod_xl.iloc[:,-1]], axis=1, keys=['DD','k'])
    elif compare == 'peak':
        data = pd.concat([mod_xl.iloc[:,list(range(0,24))].max(axis=1), mod_xl.iloc[:,-1]], axis=1, keys=['DD','k'])
        
    del mod_xl
    
    data.reset_index(inplace=True)
    data.date = data.date.astype(dt.date)#pd.to_datetime(data.date)
    data.ProfileID = data.ProfileID.astype('category')
    
    #bin daily demand into 100 equally sized bins
    if len(data.loc[data.DD==0,'DD']) > 0:
        data['int100_bins']=pd.cut(data.loc[data.DD!=0,'DD'], bins = range(0,1000,10), labels=np.arange(1, 100),
            include_lowest=False, right=True)
        data.int100_bins = data.int100_bins.cat.add_categories([0])
        data.int100_bins = data.int100_bins.cat.reorder_categories(range(0,100), ordered=True)
        data.loc[data.DD==0,'int100_bins'] = 0  
    else:
        data['int100_bins']=pd.cut(data['DD'], bins = range(0,1010,10), labels=np.arange(0, 100), include_lowest=True, right=True)
           
    #NB: use int100 for entropy calculation!
    int100_lbls = data.groupby(['k', data.int100_bins])['ProfileID'].count().unstack(level=0)
#    int100_lbls = clusterColNames(int100_lbls)
    int100_likelihood = int100_lbls.divide(int100_lbls.sum(axis=0), axis=1)
    
    if len(data.loc[data.DD==0,'DD']) > 0:
        data['q100_bins'] = pd.qcut(data.loc[data.DD!=0,'DD'], q=99, labels=np.arange(1, 100))
        data.q100_bins = data.q100_bins.cat.add_categories([0])
        data.q100_bins = data.q100_bins.cat.reorder_categories(range(0,100), ordered=True)    
        data.loc[data.DD==0,'q100_bins'] = 0
    else:
        data['q100_bins'] = pd.qcut(data['DD'], q=100, labels=np.arange(1, 101))
    
    q100_lbls = data.groupby(['k', data.q100_bins])['ProfileID'].count().unstack(level=0)
#    q100_lbls = clusterColNames(q100_lbls)
    q100_likelihood = q100_lbls.divide(q100_lbls.sum(axis=0), axis=1)
    
    return int100_likelihood.sort_index(axis=0).T, q100_likelihood.sort_index(axis=0).T


def weekdayCorr(xlabel):
    """
    """
    
    df = xlabel['k'].reset_index()
    cluster_size = df.groupby('k')['ProfileID'].count().rename('cluster_size')
    
    weekday_lbls = df.groupby(['k',df.date.dt.weekday])['ProfileID'].count().unstack(level=0)
    weekday_lbls.set_axis(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], axis=0, inplace=True)
#    weekday_lbls = clusterColNames(weekday_lbls)
    weekday_likelihood = weekday_lbls.divide(weekday_lbls.sum(axis=0), axis=1) # likelihood of assignment
    weekday_likelihood.loc['cluster_size'] = cluster_size

    return weekday_likelihood


def monthlyCorr(xlabel):
    """
    """
    
    df = xlabel['k'].reset_index()
    cluster_size = df.groupby('k')['ProfileID'].count().rename('cluster_size')
    
    month_lbls = df.groupby(['k',df.date.dt.month])['ProfileID'].count().unstack(level=0)
#    month_lbls = clusterColNames(month_lbls)
    month_likelihood = month_lbls.divide(month_lbls.sum(axis=0), axis=1)
    month_likelihood.loc['cluster_size'] = cluster_size
    
    return month_likelihood


def yearlyCorr(xlabel):
    """
    """
    
    df = xlabel['k'].reset_index()
    cluster_size = df.groupby('k')['ProfileID'].count().rename('cluster_size')
    
    year_lbls = df.groupby(['k',df.date.dt.year])['ProfileID'].count().unstack(level=0)
#    year_lbls = clusterColNames(year_lbls)
    year_likelihood = year_lbls.divide(year_lbls.sum(axis=0), axis=1)    
    year_likelihood.loc['cluster_size'] = cluster_size
    
    return year_likelihood


def daytypeCorr(xlabel):
    """
    """
    
    df = xlabel['k'].reset_index()
    cluster_size = df.groupby('k')['ProfileID'].count().rename('cluster_size')
    
    weekday_lbls = df.groupby(['k',df.date.dt.weekday])['ProfileID'].count().unstack(level=0)
    weekday_lbls.set_axis(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], axis=0, inplace=True)
    daytype = weekday_lbls[weekday_lbls.index.isin(['Monday','Tuesday','Wednesday','Thursday','Friday'])].sum(axis=0).to_frame('weekday').T
    daytype_lbls = daytype.append(weekday_lbls.loc[['Saturday','Sunday'], :])
#    daytype_lbls = clusterColNames(daytype)
    daytype_likelihood = daytype_lbls.divide(daytype_lbls.sum(axis=0), axis=1)
#    daytype_likelihood = daytype_likelihood.divide([5, 1, 1], axis=0)
    daytype_likelihood.loc['cluster_size'] = cluster_size
    
    return daytype_likelihood


def seasonCorr(xlabel):
    """
    """
    
    df = xlabel['k'].reset_index()
    cluster_size = df.groupby('k')['ProfileID'].count().rename('cluster_size')
    
    month_lbls = df.groupby(['k',df.date.dt.month])['ProfileID'].count().unstack(level=0)    
    summer = month_lbls[~month_lbls.index.isin([5, 6, 7, 8])].sum(axis=0).to_frame('summer').T
    winter = month_lbls[month_lbls.index.isin([5, 6, 7, 8])].sum(axis=0).to_frame('winter').T        
    season_lbls = summer.append(winter)
#    season_lbls = clusterColNames(season)
    season_likelihood = season_lbls.divide(season_lbls.sum(axis=0), axis=1)    
#    season_likelihood = season_likelihood.divide([8, 4], axis=0)
    season_likelihood.loc['cluster_size'] = cluster_size
    
    return season_likelihood


def saveCorr(xlabel, experiment, corr='all', n_best=1):
    """
    """
    
    if corr == 'all':    
        corr_list = ['daytype','weekday','monthly','season','yearly']
    else:
        corr_list = corr

    for corr in corr_list:
        function = corr+'Corr(xlabel)'
        df = eval(function).T  
        df['experiment'] = experiment+'BEST'+str(n_best)
        df.set_index(['experiment','cluster_size'], append=True, inplace=True)
        
        corrdir = os.path.join(data_dir, 'cluster_evaluation','k_correlations')
        os.makedirs(corrdir, exist_ok=True)
        corrpath = os.path.join(corrdir, corr+'_corr.csv')
        if os.path.isfile(corrpath):
            df.to_csv(corrpath, mode='a', index=True, header=False)
        else:
            df.to_csv(corrpath, index=True)
            
    for compare in ['total','peak']:
        corr = 'demand'
        int100, q100 = demandCorr(xlabel, compare)
        int100.columns = int100.columns.add_categories(['experiment','compare','cluster_size'])
        q100.columns = q100.columns.add_categories(['experiment','compare','cluster_size'])
        int100['experiment'] = experiment+'BEST'+str(n_best)
        int100['compare'] = compare
        int100['cluster_size'] = df.index.get_level_values(-1)
        q100['experiment'] = experiment+'BEST'+str(n_best)
        q100['compare'] = compare
        q100['cluster_size'] = df.index.get_level_values(-1)
        int100.set_index(['experiment','cluster_size','compare'], append=True, inplace=True)
        q100.set_index(['experiment','cluster_size','compare'], append=True, inplace=True)
        
        corrdir = os.path.join(data_dir, 'cluster_evaluation','k_correlations')
        os.makedirs(corrdir, exist_ok=True)
        corrpathi = os.path.join(corrdir, corr+'i_corr.csv')
        corrpathq = os.path.join(corrdir, corr+'q_corr.csv')
        if os.path.isfile(corrpathi):
            int100.to_csv(corrpathi, mode='a', index=True, header=False)
        else:
            int100.to_csv(corrpathi, index=True)
        if os.path.isfile(corrpathq):
            q100.to_csv(corrpathq, mode='a', index=True, header=False)
        else:
            q100.to_csv(corrpathq, index=True)
            
    print('Successfully saved correlation measures for ', experiment)
    return


def clusterEntropy(likelihood, threshold):
    """
    """
    
    try:
        random_likelihood = 1/len(likelihood)
    except:
        return('This function cannot compute entropy for weighted probabilities yet.')

    likelihood = likelihood.where(likelihood.cluster_size>threshold, np.nan).drop('cluster_size', axis=1) #exclude k with low membership from calculation - scew by appearing more specific than they really are
    cluster_entropy = likelihood.T.applymap(lambda x : -x*log(x,2)).sum(axis=0)
    cluster_entropy.where(cluster_entropy!=0, inplace=True)
    max_entropy = -random_likelihood*log(random_likelihood,2)*len(likelihood)
    
    return cluster_entropy, max_entropy    


def getMeasures(best_exps, threshold):
    """
    """
    
    eval_dir = os.path.join(data_dir,'cluster_evaluation')

    #Set parameters for reading evaluation data from files
    total_consE = dict()
    peak_consE = dict()
    cepath = os.path.join(eval_dir, 'consumption_error.csv')
    consumption_error = pd.read_csv(cepath, usecols=['k','experiment','compare','mape','mdape','mdlq','mdsyma']).drop_duplicates(subset=['k','experiment','compare'], keep='last').set_index('k', drop=True)
    consumption_error.rename({'experiment':'experiment_name'}, axis=1)
    
    peak_coincR = dict()
    pcrpath = os.path.join(eval_dir, 'peak_coincidence.csv')
    peak_eval = pd.read_csv(pcrpath).drop_duplicates(subset=['k','experiment'], keep='last').set_index('k', drop=True)
    
    temporal_entropy = dict(zip(best_exps, [dict()]*len(best_exps)))   
    corr_path = os.path.join(data_dir, 'cluster_evaluation', 'k_correlations')
    temp_files = ['yearly', 'weekday', 'monthly']

    demand_entropy = dict(zip(best_exps, [dict()]*len(best_exps)))
    compare = ['total','peak']
    
    #Generate evaluation measures for each experiment in best experiments list
    for e in best_exps:
        #total consumption error
        consE = consumption_error.loc[(consumption_error.experiment==e)&(consumption_error.compare=='total'),:]
        total_consE[e] = {'mape':consE.mape,'mdape':consE.mdape,'mdlq':consE.mdlq,'mdsyma':consE.mdsyma}     
        #peak consumption error
        consE = consumption_error.loc[(consumption_error.experiment==e)&(consumption_error.compare=='peak'),:]
        peak_consE[e] = {'mape':consE.mape,'mdape':consE.mdape,'mdlq':consE.mdlq,'mdsyma':consE.mdsyma}
        #peak coincidence ratio
        peak_coincR[e] = {'coincidence_ratio': peak_eval.loc[peak_eval['experiment']==e,'coincidence_ratio']}

        #temporal entropy
        te = dict()
        for temp in temp_files:
            df = pd.read_csv(os.path.join(corr_path, temp+'_corr.csv'), header=[0]).drop_duplicates(
                    subset=['k','experiment'], keep='last').set_index('k', drop=True) 
            likelihood = df[df.experiment == e+'BEST1'].drop('experiment', axis=1)
            entropy, maxE = clusterEntropy(likelihood, threshold)
            te[temp+'_entropy'] = entropy#.reset_index(drop=True)
        temporal_entropy.update({e:te})      
        
        #demand entropy
        co = dict()
        df_temp = pd.read_csv(os.path.join(corr_path, 'demandi_corr.csv'), header=[0]).drop_duplicates(
                subset=['k','experiment','compare'], keep='last').set_index('k', drop=True)
        for c in compare:
            likelihood = df_temp[(df_temp.experiment == e+'BEST1')&(df_temp.compare==c)].drop(['experiment',
                                 'compare'], axis=1)
            entropy, maxE = clusterEntropy(likelihood, threshold)
            co[c+'_entropy'] = entropy#.reset_index(drop=True)
        demand_entropy.update({e:co})
    
    return total_consE, peak_consE, peak_coincR, temporal_entropy, demand_entropy


def saveMeasures(best_exps, threshold):
    """
    """
    
    measures = zip(['total_consE', 'peak_consE', 'peak_coincR', 'temporal_entropy', 
                    'demand_entropy'], getMeasures(best_exps, threshold))
    mean_measures = list()

    for m, m_data in measures:
        for k,v in m_data.items():
            for i,j in v.items():
                me = meanError(j)
                mean_measures.append([m, k, i, me])   
            
    evaluation_table = pd.DataFrame(mean_measures,
    					columns=['measure','experiment','metric','value'])
    evalcrit = evaluation_table.measure.apply(lambda x: x.split('_',1)[1])
    evaluation_table.insert(0, 'evaluation_criteria', evalcrit)

    complete_eval = evaluation_table.pivot_table(index=[
            'evaluation_criteria','measure','metric'], columns='experiment').reset_index()
    complete_eval.T.to_csv(
            os.path.join(data_dir,'cluster_evaluation','cluster_entropy.csv'), 
            index=True, header=False)
    
    return complete_eval.T


def householdEntropy(xlabel):
    """
    """
    
    label_data = xlabel['k']
    
    df = label_data.reset_index()
    
    data = df.groupby(['ProfileID','k'])['date'].count().rename('day_count').reset_index()
    hh_lbls = data.pivot(index='ProfileID',columns='k',values='day_count')
    hh_likelihood = hh_lbls.divide(hh_lbls.sum(axis=1), axis=0)
    random_likelihood = 1/47
    
    cluster_entropy = hh_likelihood.applymap(lambda x : -x*log(x,2)).sum(axis=1)
    max_entropy = -random_likelihood*log(random_likelihood,2)*47
    
    return cluster_entropy, max_entropy


def monthlyHHE(lbls, S, month_ix):
    """
    """
        
    hhe, me = householdEntropy(lbls[lbls.date.dt.month==
    			month_ix].set_index(['ProfileID','date']))
    Sent = pd.concat([S, (hhe/me)], axis=1, join='inner').rename(columns={0:'rele'})
    sg = Sent.groupby('monthly_income').aggregate({'rele':['mean','std']})

    return sg


def clusterReliability(xlabel):
    """
    """
    
    kxl = xlabel.groupby(by='ProfileID', level=0)['k'].value_counts().reset_index(name='k_count')
    maxkxl = kxl.iloc[kxl.groupby('ProfileID')['k_count'].idxmax()]
    he = householdEntropy(xlabel)[0].reset_index(name='entropy')
    cr = pd.merge(maxkxl, he, on='ProfileID', sort=True)
    cr_out = cr.groupby('k').agg({'k_count':'count', 'entropy':'mean'}).rename(columns={'k_count':'hh_count'})
    cr_out['stdev'] = xlabel.groupby('k').std().mean(axis=1)
    cr_out['daily_demand'] = xlabel.groupby('k').mean().sum(axis=1)
        
    return cr_out


def householdReliability(xlabel):
    """
    This function does not take cognisance of years 
    - ie clusters are aggregated over the entire duration of observation.
    """
    
    kxl = xlabel.groupby(by='ProfileID', level=0)['k'].value_counts().reset_index(name='k_count')    
    maxkxl = kxl.iloc[kxl.groupby('ProfileID')['k_count'].idxmax()]

    elec_bins = xlabel[['elec_bin','k']].drop_duplicates(subset=['elec_bin','k']).reset_index(drop=True)
    elec_bins_dict = dict(zip(elec_bins['k'], elec_bins['elec_bin']))

    maxkxl.loc[:,'elec_bin'] = maxkxl.loc[:,'k'].map(elec_bins_dict)
    
    he = householdEntropy(xlabel)[0].reset_index(name='entropy')
    hr = pd.merge(maxkxl, he, on='ProfileID', sort=True).set_index('ProfileID')
    hr['stdev'] = xlabel.loc[:,'0':'23'].groupby('ProfileID').std().mean(axis=1)
    hr['daily_demand'] = xlabel.loc[:,'0':'23'].groupby('ProfileID').mean().sum(axis=1)
    
    idload = surveys.loadID()
    ids = idload.loc[idload['Unit of measurement']==2,['ProfileID','Year','Municipality']].set_index('ProfileID')
    
    hr_out = pd.merge(hr, ids, on='ProfileID')
        
    return hr_out
