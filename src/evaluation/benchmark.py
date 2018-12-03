#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 13:33:32 2017

@author: saintlyvi
"""

import pandas as pd    
import numpy as np
import os
from math import ceil

import colorlover as cl
import plotly.offline as offline
import plotly.graph_objs as go
import plotly as py
offline.init_notebook_mode(connected=True) #set for plotly offline plotting

from support import dpet_dir, image_dir

def bmDemandSummary(model_dir = dpet_dir):
    """
    Retrieves demand summary expert model stored in csv files in model_dir. File names must end with '_summary.csv' .
    """
    try:
        files = os.listdir(model_dir) #get data files containing expert model
        summaryfiles = [f for f in files if "summary" in f]
        
        summary = pd.DataFrame()
        for s in summaryfiles:
            name = s.split('_summary.csv')[0]
            data = pd.read_csv(os.path.join(model_dir, s))
            data['class'] = name
            summary = summary.append(data)    
        summary.reset_index(inplace=True, drop=True)
        summary.rename(columns={'Year':'YearsElectrified'}, inplace=True)
        
    except:
## TODO: get data from energydata.uct.ac.za        
        pass
    
    return summary

def bmHourlyProfiles(model_dir = dpet_dir):
    """
    Retrieves hourly profiles expert model stored in csv files in model_dir. File names must end with '_hourly.csv' .
    """
    try:
        files = os.listdir(model_dir) #get data files containing expert model
        hourlyfiles = [f for f in files if "hourly" in f]
        
        hourlyprofiles = pd.DataFrame()
        for h in hourlyfiles:
            name = h.split('_hourly.csv')[0]
            data = pd.read_csv(os.path.join(model_dir, h))
            data['class'] = name
            hourlyprofiles = hourlyprofiles.append(data)    
        hourlyprofiles.reset_index(inplace=True, drop=True)
        hourlyprofiles.rename(columns={'Year':'YearsElectrified', 
                                       'Month':'month', 
                                       'Day Type':'daytype', 
                                       'Time of day [hour]':'hour',}, inplace=True)
        hourlyprofiles.month = hourlyprofiles.month.astype('category')
        hourlyprofiles.month.cat.rename_categories({
                'April':4, 'August':8, 'December':12, 'February':2, 'January':1, 
                'July':7, 'June':6, 'March':3, 'May':5, 'November':11, 
                'October':10, 'September':9}, inplace=True)
## TODO
    #convert month string to int 1-12
        
    except:
## TODO: get data from energydata.uct.ac.za        
        pass
    
    return hourlyprofiles

def benchmarkModel():
    """
    Fetch data for existing/expert DPET model.
    """
    hp = bmHourlyProfiles()
    ds = bmDemandSummary()
    dsts = 'Energy [kWh]'
    hpts = 'Mean [kVA]'
    
    return ds, hp, dsts, hpts


def plotBmDemandSummary(customer_class, model_dir = dpet_dir):
    """
    This function plots the average monthly energy consumption for a specified customer class from 
    1 to 15 years since electrification. Data is based on the DPET model.
    """
    
    summary = bmDemandSummary(model_dir)
    df = summary[summary['class']==customer_class][['YearsElectrified','Energy [kWh]']] 
    data = [go.Bar(
                x=df['YearsElectrified'],
                y=df['Energy [kWh]'],
                name=customer_class
            )]
     
    layout = go.Layout(
                title='Annualised Monthly Energy Consumption for "' + customer_class + '" Customer Class',
                xaxis=dict(
                    title='years since electrification',
                    tickfont=dict(
                    size=14,
                    color='rgb(107, 107, 107)'
                    )
                ),
                yaxis=dict(
                    title='average annual kWh/month',
                    titlefont=dict(
                        size=16,
                        color='rgb(107, 107, 107)'
                    )
                )
    )
    
    return offline.iplot({"data":data, "layout":layout}, filename=os.path.join(image_dir,'demand_summary_'+customer_class+'.png'))

def plot15YearBmDemandSummary(model_dir = dpet_dir):
    """
    This function plots the average monthly energy consumption for all customer classes from
    1 to 15 years since electrification. Data is based on the DPET model.
    """
    
    clrs = ['Greens','RdPu','Blues','YlOrRd','Purples','Reds', 'Greys']
    
    summary = bmDemandSummary(model_dir)
    df = summary[['class','YearsElectrified','Energy [kWh]']].sort_values(by='Energy [kWh]')
    data = []
    
    count=0
    for c in df['class'].unique():

        trace = go.Scatter(
                x=df.loc[df['class'] == c, 'YearsElectrified'],
                y=df.loc[df['class'] == c, 'Energy [kWh]'],
                name=c,
                fill='tonexty',
                mode='lines',
                line=dict(color=cl.flipper()['seq']['3'][clrs[count]][1], 
                                     width=3)
        )
        data.append(trace)
        count+=1
        
    layout = go.Layout(
                title='Annualised Monthly Energy Consumption for Domestic Energy Consumers',
                xaxis=dict(
                    title='years since electrification',
                    tickfont=dict(
                    size=14,
                    color='rgb(107, 107, 107)'
                    )
                ),
                yaxis=dict(
                    title='average annual kWh/month',
                    titlefont=dict(
                        size=16,
                        color='rgb(107, 107, 107)'
                    )
                ),
    )
    
    return offline.iplot({"data":data, "layout":layout}, filename=os.path.join(image_dir,'15year_demand_summary'+'.png'))

def plotBmHourlyHeatmap(customer_class, year_list, daytype='Weekday', model_dir=dpet_dir):
    """
    This function plots the hourly load profiles for a specified customer class, day type and list of years since electrification. Data is based on the DPET model.
    """
    
    df = bmHourlyProfiles(model_dir)
    maxdemand = df['Mean [kVA]'].max() #get consistent max demand & color scale across classes
    df = df[(df['daytype']==daytype) & (df['class']==customer_class)]
    
    #set heatmap colours
    colors = cl.flipper()['div']['5']['RdYlBu']
    scl = [[0,colors[0]],[0.25,colors[1]],[0.5,colors[2]],[0.75,colors[3]],[1,colors[4]]]
    
    #set subplot parameters
    if len(year_list) < 3:
        ncol = len(year_list)
    else:
        ncol = 3
    nrow = ceil(len(year_list)/ncol)
    fig = py.tools.make_subplots(rows=nrow, cols=ncol, 
                    subplot_titles=['Year ' + str(x) for x in year_list], 
                    horizontal_spacing = 0.1, print_grid=False)
    r = 1 #initiate row
    c = 1 #initiate column
    
    for yr in year_list:
        if c == ncol + 1: 
            c = 1
        ro = ceil(r/ncol)
        
        #set colorbar parameters
        if nrow == 1:
            cblen = 1
            yanc = 'middle'
        else:
            cblen = 0.5
            yanc = 'bottom'
        
        if r == 1: #toggle colorscale
            scl_switch=True
        else:
            scl_switch=False
        
        #generate trace
        try:
            data = df[df['YearsElectrified']==yr]
            z = data['Mean [kVA]'].reset_index(drop=True)
            x = data['hour']
            y = data.month
            hovertext = list()
            for yi, yy in enumerate(y.unique()):
                hovertext.append(list())
                for xi, xx in enumerate(x.unique()):
                    hovertext[-1].append('hour: {}<br />month: {}<br />{:.3f} kVA'.format(xx, yy, z[24 * yi + xi]))
            trace = go.Heatmap(z = z, 
                               x = x, 
                               y = y,
                               zmin = 0,
                               zmax = maxdemand,
                               text = hovertext,
                               hoverinfo="text",
                               colorscale=scl,
                               reversescale=True,
                               showscale=scl_switch,
                               colorbar=dict(
                                           title='kVA',
                                           len=cblen,
                                           yanchor=yanc))
            fig.append_trace(trace, ro, c)
            
        except:
            pass
        
        c += 1
        r += 1

    fig['layout'].update(showlegend=False, 
      title='<b>'+customer_class+'</b> mean estimated <b>'+daytype+'</b> energy demand (kVA) <br />' + ', '.join(map(str, year_list[:-1])) + ' and ' + str(year_list[-1]) + ' years after electrification',
      height=350+300*(nrow-1))
    
    for k in range(1, len(year_list)+2):
          fig['layout'].update({'yaxis{}'.format(k): go.YAxis(type = 'category',
                                                              ticktext = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],#data.month.unique(),
                                                              tickvals = np.arange(1, 13, 1),
                                                              tickangle = -15,
                                                              tickwidth = 0.5
                                                             ),
                                
                                'xaxis{}'.format(k): go.XAxis(title = 'Time of day (hours)', 
                                                              tickvals = np.arange(0, 24, 2))
                                })    
                                
    return offline.iplot(fig, filename='testagain') 


def plotBmHourlyProfiles(electrified_min, electrified_max, customer_class, model_dir=dpet_dir, title=''):

    hourlyprofiles = bmHourlyProfiles()
    hourlyprofiles['season'] = hourlyprofiles['month'].apply(lambda x: 'winter' if x in [6, 7, 8, 9] else 'summer')
    electrified_range = range(electrified_min+1,electrified_max+1)

    df = hourlyprofiles[(hourlyprofiles['class']==customer_class)&(hourlyprofiles['YearsElectrified'].isin(electrified_range)) ].groupby(['season','daytype','hour'])['Mean [kVA]'].mean().unstack()/0.23 #get mean profile values and % by 0.23 to get from kVA to A                                           
                          
    experiment_name = 'benchmark_'+customer_class+'_'+str(electrified_min)+'-'+str(electrified_max)+'yrs_electrified'
    
    if title == '':
        plot_title = experiment_name
    else:
        plot_title = title
   
    data = []  

   #Create colour scale
    spectral = cl.scales['11']['div']['Spectral']
    colours = [spectral[c] for c in [0,2,3]] + [spectral[c] for c in [8,9,10]]

    i = 0
    for col in df.T.columns:
        trace = go.Scatter(
            x = df.T.index.map(lambda x: str(x)+'h00'),
            y = df.T.iloc[:,i],
            line = {'color':colours[i],'width':3},
            mode = 'lines',
            name = col[0]+': '+col[1]
        )
        data.append(trace)
        i+=1
    
    fig = go.Figure(data=data, layout= go.Layout(title=plot_title, height=400, font=dict(size=20)))
    
    fig['layout']['xaxis'].update(title='time of day', dtick=2, titlefont=dict(size=18), tickfont=dict(size=16))
    fig['layout']['yaxis'].update(title='hourly electricity demand (A)', titlefont=dict(size=18), tickfont=dict(size=16))
    fig['layout']['margin'].update(t=50,r=80,b=100,l=90,pad=10),
#    fig['layout']['title']['font'] = dict(size=20)
    
    offline.plot(fig, filename='img/benchmark/bm0/'+experiment_name+'.html')
