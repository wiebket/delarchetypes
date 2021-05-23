#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 13:59:37 2017

@author: Wiebke Toussaint
"""

import pandas as pd
import os
import matplotlib.pyplot as plt

from delprocess.surveys import genS
from ..cluster.results import getLabels, getExpDetails

from ..support import image_dir, data_dir

def plotF(F, columns, save_name=None):
    """
    Plots column category counts in F. Columns must be a list.
    """
    if save_name is None:
        dir_path = os.path.join(image_dir, 'experiment')
    else:
        dir_path = os.path.join(image_dir, 'experiment', save_name)
    os.makedirs(dir_path, exist_ok=True)
    
    data = F[['ProfileID']+columns]

    if len(columns) == 1:
        fig, axes = plt.subplots(nrows=1, ncols=1)
        data.groupby(columns).ProfileID.count().plot(
                'bar', title='Count of observations in ' + columns[0] + ' bins', figsize=(10, 6))
        plt.tight_layout()
        fig.savefig(os.path.join(dir_path, columns[0]+'.png'))
        plt.close()
    
    elif len(columns) > 1:
        fig, axes = plt.subplots(nrows=len(data.columns)-1, ncols=1)
        r = 0
        for c in data.columns.drop('ProfileID'):
            data.groupby(c).ProfileID.count().plot('bar', 
                        title='Count of observations in ' + c + ' bins', 
                        ax=axes[r], figsize=(10-len(data.columns)/4, len(data.columns)*4))
            r += 1
            plt.tight_layout()            

        fig.savefig(os.path.join(dir_path, save_name+'.png'))
        plt.close()
    
    return fig

def genFProfiles(experiment, socios, n_best=1, keep_years=False, savefig=False):
    """
    generates a socio-demographic feature set
    """
    
    year_start, year_end, drop_0, prepro, bin_X, exp_root = getExpDetails(experiment)
    
    kf_dir = os.path.join(
            data_dir, 'cluster_evaluation','k_features', experiment+'_'+socios+'BEST'+str(n_best))
    kf_path = kf_dir+'.csv'
    if os.path.exists(kf_path) is True:
        F = pd.read_csv(kf_path, low_memory=False)

    else:
        os.makedirs(kf_dir, exist_ok=True)
        print('Extracting and creating feature data...')
        # Get cluster labels
        XL = getLabels(experiment, n_best)
        XL = XL.drop(columns=[str(i) for i in range(0,24)], axis=0).reset_index()
    
        # Add temporal features
        XL['year']=XL.date.dt.year
        XL['month']=XL.date.dt.month_name()
        XL['weekday']=XL.date.dt.weekday_name
        
        winter = ['May','June','July','August']
        work_week = ['Monday','Tuesday','Wednesday','Thursday']
        
        XL['season'] = XL.month.where(XL.month.isin(winter), 'summer')
        XL['season'] = XL.season.where(XL.season=='summer', 'winter')
        XL['daytype'] = XL.weekday.where(~XL.weekday.isin(work_week), 'weekday')

        if keep_years == True:
            kXL = XL.groupby(['ProfileID','year','season','daytype']
                                )['k'].value_counts().reset_index(name='k_count')
        else:
            kXL = XL.groupby(['ProfileID','season','daytype']
                                )['k'].value_counts().reset_index(name='k_count')
    
        kXL = kXL[kXL.k_count>1] #keep rows with two or more occurences of k
        
        S = genS(socios, year_start, year_end, 'feather').reset_index()  
        
        #merge socio-demographic, geographic and temporal features
        F = pd.merge(kXL, S, how='inner',on='ProfileID')
        del XL, S
        
        columns = F.columns.tolist()
        columns.remove('k')
        F = F[columns + ['k']]
        
        if keep_years == True:
            for y in range(year_start, year_end+1):
                #for c in F.columns:
                    #F.loc[:,c] = F.loc[:,c].astype('category')
                Y = F[F.year==y]
                Y.drop(columns=['ProfileID','year'], inplace=True)
                Y.to_csv(os.path.join(kf_dir, str(y)+'.csv'), index=False)
            print('Saved feature sets for years')
            
        else:
            F.to_csv(kf_path, index=False)
            print('Saved feature set to', kf_path)
    
    if savefig is True:
        for c in F.columns:#.drop(['ProfileID']):
            plotF(F, [c], socios)
    
    return F

def describeFProfiles(experiment):
    
    F = genFProfiles(experiment, 'features4')
    cats = dict(
            cb_size = ["<20","21-60",">61"],
            floor_area = ['0-50', '50-80', '80-150', '150-250','250-800'],
            years_electrified = ["0-5yrs", "5-10yrs", "10-15yrs", "15+yrs"],
            monthly_income = ["R0-R1799","R1800-R3199","R3200-R7799","R7800-R11599",
                              "R11600-R19115","R19116-R24499","R24500-R65499","+R65500"]
    )
       
    for c, v in cats.items():
        F[c] = F[c].astype('category')
        F[c].cat.reorder_categories(v, ordered=True,inplace=True)
    
    sf = pd.DataFrame()
    for c in F.columns[1:-2]:
        sample = pd.melt(F.groupby(c)['k_count'].count().reset_index(), col_level=0,
                         id_vars=['k_count'], value_vars=[c])
        sf = sf.append(sample)
    
    mapper = dict(season='temporal',
              daytype='temporal', 
              Province='spatial', 
              adults = 'occupants',
              children = 'occupants',
              pension = 'occupants',
              unemployed = 'occupants',
              part_time = 'occupants',
              monthly_income = 'economic', 
              geyser = 'appliances',
              floor_area = 'dwelling',
              water_access = 'dwelling',
              wall_material = 'dwelling',
              roof_material = 'dwelling',
              cb_size = 'connection',
              years_electrified = 'connection') 
    
    sf['category'] = sf['variable'].apply(lambda x: mapper[x])
    
    sf.set_index(['category','variable','value'], inplace=True)
    sf.columns = ['sample_count (unweighted)']
    sf.sort_index(level=[0,1], ascending=False, sort_remaining=False, inplace=True)

    sf.rename(index={'pension': 'pensioners', 'part_time':'part_time_employed', 
                     'floor_area':'floor_area (m^2)'}, inplace=True)
        
    return sf.sort_index(level=[0], ascending=False, sort_remaining=False)

def genArffFile(experiment, socios, filter_features=None, skip_cat=None, weighted=True, n_best=1):
    
    kf_name = experiment+'_'+socios+'BEST'+ str(n_best)
    kf_dir = os.path.join(data_dir, 'cluster_evaluation','k_features', kf_name)
    os.makedirs(kf_dir, exist_ok=True)   
    
    F = genFProfiles(experiment, socios, n_best)
    F.drop('ProfileID', axis=1, inplace=True)
    
    for col in ['water_access','wall_material','roof_material']:
        F[col] = F[col].str.replace(' ','_')
    
    if filter_features != None:
        apply_filter = eval(filter_features)
        for k,v in apply_filter.items():
            Ftemp = F[F[k]==v]
            F = Ftemp.drop(k, axis=1)
            kf_name += '_' + v
        kf_dir = os.path.join(kf_dir, kf_name)
   
    
    if weighted == True:
        kf_path = kf_dir+'.arff'
    elif weighted == False:
        kf_path = kf_dir+'noW.arff'
    
    attributes = [] 
    for c in F.columns:
        if type(skip_cat) is list:
            if c == 'k_count':
                pass
            elif c in skip_cat:
                    att = '@attribute ' + c + ' numeric'
                    attributes.append(att)
            else:
                att = '@attribute ' + c
                cats = F[c].astype('category')
                att += ' {'+",".join(map(str, cats.cat.categories))+'}'
                attributes.append(att)
        else:
            if c == 'k_count':
                pass
            else:
                att = '@attribute ' + c
                cats = F[c].astype('category')
                att += ' {'+",".join(map(str, cats.cat.categories))+'}'
                attributes.append(att)

    F.fillna('?', inplace=True)
            
    with open(kf_path, 'a+') as myfile:
        myfile.write('@relation ' + kf_name + '\n\n')
        for a in attributes:  
            myfile.write(a+'\n')
        myfile.write('\n@data\n')        
        for r in F.iterrows(): 
            if weighted == True:
                weight = r[1]['k_count']
            elif weighted == False:
                weight = ''
            vals = r[1].drop('k_count')
            myfile.write(','.join(map(str,vals)) + ',{'+str(weight)+'}\n')

    return print('Successfully created',experiment, socios, 'arff file.')

def genFHouseholds(experiment, socios, n_best=1):
    
    F = genFProfiles(experiment, socios, n_best, savefig=False)  
    Fhh = F.iloc[F.groupby(['ProfileID','season','daytype'])['k_count'].idxmax()]     
    
#    Fsub.to_csv(kf_path, index=False)
    
    return Fhh


def checkFeatures(data, appliances):
    """
    This function error checks appliance features for records that indicate appliance usage but no ownership.
    """
    
    err = pd.DataFrame()
    for a in appliances:
        try:
            e = data.loc[(data[a]==0)&(data[a+'_use']>0), [a,a+'_use',a+'_broken']]
            print(e)
            err = err.append(e)
        except:
            pass
        
    return err