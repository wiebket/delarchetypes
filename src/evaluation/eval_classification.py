#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 09:52:39 2018

@author: saintlyvi
"""

import pandas as pd
import os
from glob import glob

from support import data_dir, results_dir, experiment_dir

def joinResults(searchterm):
    mod = pd.DataFrame()
    p = os.path.join(results_dir,'classification_results')
    for file in os.listdir(p): 
        if searchterm in file: 
            data = pd.read_csv(os.path.join(p, file))
    #        print(mod.columns.difference(data.columns))
            mod = pd.concat([mod, data], axis=0, sort=True)
            print(file)
    mod.sort_values(by=['Key_Dataset','Key_Run','Key_Scheme_options'], inplace=True)
    mod['Key_Scheme_options'] = mod['Key_Scheme_options'].apply(lambda x:x.replace('\\"',''))
    
    mod.to_csv(os.path.join(p, 'classification_'+searchterm+'.csv'), index=False)
    
    return print('Results collated and saved.')
    
def formatResults(filename='classification_output', cats=['default+CGD', 'K2-P1','K2-P2','K2-P3','K2-P4','HC-P1','HC-P2','HC-P3','HC-P4','BestFirst']):
    df = pd.read_csv(os.path.join(results_dir,'classification_results',filename+'.csv'),
                     usecols=['Key_Dataset', 'Key_Fold','Key_Run', 'Key_Scheme',
                              'Key_Scheme_options','Percent_correct'])
    df['Key_Scheme'] = df['Key_Scheme'].apply(lambda x: x.split('.')[-1])
    df['Experiment'] = df['Key_Dataset'].apply(lambda x: '_'.join(x.split('_')[:2])+' '+'_'.join(x.split('_')[2:4]))
    df['Features'] = df['Key_Dataset'].apply(lambda x: x.split('_')[-1].split('BEST')[0])
    df.loc[:,'Key_Scheme_options'] = df.Key_Scheme_options.astype('category')
    df.Key_Scheme_options.cat.rename_categories(cats, inplace=True)
    df.rename({'Key_Scheme_options':'Options'}, axis=1, inplace=True)

    percent_correct = round(df.pivot_table(index=['Experiment','Features'], columns=['Key_Scheme','Options'], values=['Percent_correct'], aggfunc='mean'), 2).fillna('')
    
    return df, percent_correct
