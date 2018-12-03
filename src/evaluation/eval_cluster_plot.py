#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 09:31:48 2018

@author: saintlyvi
"""

import pandas as pd
import numpy as np
import os

import plotly.plotly as py
import plotly.offline as po
import plotly.graph_objs as go
import plotly.tools as tools
import colorlover as cl
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
cf.go_offline()

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

import evaluation.eval_clusters as ec
from support import data_dir, image_dir

clustering_evaluation_dir = os.path.join(data_dir, 'cluster_evaluation', 'plots', 'clustering_evaluation')
cluster_analysis_dir = os.path.join(data_dir, 'cluster_evaluation', 'plots', 'cluster_analysis')

def plotPrettyColours(data, grouping):
    
    #make directories for saving plot htmls
    os.makedirs(clustering_evaluation_dir, exist_ok=True)
    os.makedirs(cluster_analysis_dir, exist_ok=True)
   
    if grouping == 'experiments':
        colour_seq = ['Reds','Oranges','YlOrBr','YlGn','Greens','BuGn','Blues','RdPu',
                      'PuBu','Purples','PuRd','YlGnBu','YlOrRd']
        df = pd.DataFrame(data.experiment_name.unique(), columns=['name'])
        df['root'] = df.applymap(lambda x: '_'.join(x.split('_',2)[0:2]))
        
    elif grouping == 'elec_bin':
        colour_seq = ['YlGn','PuRd','Blues','YlOrBr','Greens','RdPu','Oranges',
                      'Purples','PuBu','BuGn','Reds']
        df = data['elec_bin'].reset_index().rename({'elec_bin':'root', 'k':'name'}, axis=1)

    df['root'] = df.root.astype('category')                
    df.root.cat.rename_categories(colour_seq[:len(df.root.cat.categories)], inplace=True)
    col_temp = df.groupby('root').apply(lambda x: len(x))
    
    my_cols = list()
    for c, v in col_temp.items():
        try:
            i = 0
            gcol=list()
            while i < v:
                gcol.append(cl.scales['9']['seq'][c][2+i])
                i+=1
        except:
            i = 0
            gcol=list()
            jump = int(80/v)
            while i < v:
                gcol.append(cl.to_rgb(cl.interp(cl.scales['9']['seq'][c], 100))[-1-jump*i])
                i+=1
        my_cols+=gcol
    
    colours = dict(zip(df.name, my_cols))
    
    return colours

def plotClusterIndex(index, title, experiments, threshold=1200, groupby='algorithm', ylog=False):
    
    cluster_results = ec.readResults()
    cluster_results = cluster_results[cluster_results.total_sample>threshold]
    cluster_results = cluster_results.groupby(['experiment_name','som_dim','n_clust']).mean().reset_index() 
    cluster_results = cluster_results[cluster_results.experiment_name.isin(experiments)]
    cluster_results['series'] = cluster_results['som_dim'].where(
         (cluster_results['n_clust'] != 0) & 
         (cluster_results['som_dim'] != 0), '')
    
    colours = plotPrettyColours(cluster_results,'experiments')
    df = cluster_results.set_index(['experiment_name','series'])[[index,'clusters']]
    data = pd.pivot_table(df[[index,'clusters']], index='clusters', columns=df.index, values=index)

    #generate plot data
    groupdict = dict(zip(['experiment','algorithm'],[0,1]))
    traces = []
    for c in data.columns:
        t = c[0].split('_',1)+[str(c[1])]
        x = data.index
        y = data[c]
        n = ' '.join(t)
        hovertext = list()
        for i in x:
            hovertext.append('{}<br />{}: {:.2f}<br />{:.0f} clusters<br />'.format(n, index, y[i], i))

        traces.append(dict(
            x=x,
            y=y,
            name=n,
            legendgroup=t[groupdict[groupby]],
            mode='lines+markers',
            marker=dict(size=3),
            line=dict(color=colours[c[0]]),
            text = hovertext,
            hoverinfo='text',
            connectgaps=True
        ))

    #set layout
    if ylog == True:
        yax = dict(title = index+' (log scale)' , type='log')
    else:
        yax = dict(title = index)
    layout = go.Layout(
            title= title,
            margin=go.Margin(t=50,r=50,b=50,l=50, pad=10),
            height= 700,
            xaxis=dict(title = 'n clusters (log scale)', type='log'),
            yaxis=yax,
            hovermode = "closest"
            )

    fig = {'data':traces, 'layout':layout }
    return po.plot(fig, filename=clustering_evaluation_dir+'/cluster_index_'+index+'.html')

def plotClusterCentroids(centroids, groupby='_bin', n_best=1, title=''):

    n_best = centroids['n_best'].unique()[0]
    experiment_name = centroids['experiment'].unique()[0]
    
    if len(centroids.elec_bin.unique()) == 1: 
        centroids = ec.rebinCentroids(centroids)
    if 'bin' in centroids.elec_bin.unique()[0]:
        centroids = ec.renameBins(centroids, centroids)
    
    traces = centroids.iloc[:, 0:24].T
    n_clust = len(traces.columns)
    largest = 'cluster '+str(centroids.cluster_size.idxmax())
    
    if title == '':
        plot_title = 'cluster profiles '+experiment_name+' (n='+str(n_clust)+') TOP '+str(n_best)
    else:
        plot_title = title
        
    colours =  plotPrettyColours(centroids, 'elec_bin')    
    fig = tools.make_subplots(rows=3, cols=1, shared_xaxes=False, specs=[[{'rowspan': 2}],[None],[{}]],
                              subplot_titles=[plot_title,''], print_grid=False)  

    legend_group = centroids['elec_bin'].reset_index()
    
    if groupby == '_bin':
        i = 0
        traces.columns = ['cluster ' + str(k) for k in traces.columns.values]   
        for col in traces.columns:
            if col == largest:
                width = 4
            else:
                width = 2
            fig.append_trace({'x': traces.index+'h00', 'y': traces[col], 
                              'line':{'color':colours[legend_group['k'][i]],'width':width}, 
                              'type': 'scatter', 'legendgroup':legend_group['elec_bin'][i], 
                              'name': col}, 1, 1)
            i+=1
        for b in centroids['elec_bin'].unique():
            t = centroids.cluster_size[centroids.index[centroids.elec_bin==b]]
            fig.append_trace({'x': ['cluster '+str(v) for v in t.index.values],'y':t.values,'type':'bar','name': b,
                        'legendgroup':b,'marker':dict(color=[colours[k] for k in t.index.values])},3,1)
        
    else:
        groupby = ''
            #Create colour scale
        paired = cl.scales['12']['qual']['Paired']
        spectral = cl.scales['11']['div']['Spectral'][0:5] + cl.scales['11']['div']['Spectral'][6:]
        set3 = cl.scales['12']['qual']['Set3'][2:]
        colours = spectral + [paired[i] for i in [1,3,5,7,9,11,0,2,4,6,8]]  +spectral + set3
        colours = colours*5
        i = 0
        for col in traces.columns.sort_values():
            fig.append_trace({'x': traces.index+'h00', 'y': traces[col], 
                              'line':{'color':colours[i],'width':2},
                              'legendgroup':col, 'type': 'scatter', 'name': 'cluster '+str(col)}, 1, 1)
            fig.append_trace({'x':['cluster '+str(col)],'y':[centroids.loc[col,'cluster_size']],
                                    'type':'bar','legendgroup':col,'name':'cluster '+str(col),
                                    'showlegend': False,
                                    'marker': {'color':colours[i]}}, 3, 1)
            i+=1
    
    fig['layout']['xaxis1'].update(title='time of day', dtick=2, titlefont=dict(size=18), tickfont=dict(size=16))
    fig['layout']['xaxis2'].update(tickfont=dict(size=16))
    fig['layout']['yaxis1'].update(title='hourly electricity demand (A)', titlefont=dict(size=18), tickfont=dict(size=16))
    fig['layout']['yaxis2'].update(title='cluster size (# members)', titlefont=dict(size=18), tickfont=dict(size=16))
    fig['layout']['margin'].update(t=50,r=80,b=100,l=90,pad=10),
    fig['layout'].update(height=700, hovermode = "closest", grid=dict(ygap=0.35, xgap=0.35))
    fig['layout']['annotations'][0]['font'] = dict(size=20)
#    for i in fig['layout']['annotations']:
#        i['font'] = dict(size=20) #set subplot title size
    
    po.plot(fig, filename=cluster_analysis_dir+'/cluster_centroids'+groupby+'_'+experiment_name+'_'+title+'.html')
    
def plotClusterLabels(label_data, year, n_clust=None, som_dim=0):
    
#    if n_clust is None:
#        c = label_data.columns[0]
#    else:
#        c = str(som_dim)+'_'+str(n_clust)
    df = label_data.loc[pd.IndexSlice[:,str(year)],'k'].reset_index()
    df.date = df.date.dt.date
    
    fig = df.iplot(kind='heatmap', title='Daily cluster labels for profiles in '+str(year), x='date', y='ProfileID', z='k', colorscale='spectral', asFigure=True)

    fig['layout']['yaxis'].update(dict(type='category',title='ProfileID'))
    fig['layout']['xaxis'].update(dict(title='Date'))
    for i, trace in enumerate(fig['data']):
        hovertext = list()
        for j in range(len(trace['x'])):
            hovertext.append('date: {}<br />cluster label: {}<br />ProfileID: {}<br />'.format(trace['x'][j], trace['z'][j]+1, trace['y'][j]))
        trace['text'] = hovertext
        trace['hoverinfo']='text'
    
    return po.iplot(fig)

#---------------------------
# Prepping the colorbar
#---------------------------

def display_cmap(cmap): #Display  a colormap cmap
    plt.imshow(np.linspace(0, 100, 256)[None, :],  aspect=25, interpolation='nearest', cmap=cmap) 
    plt.axis('off')
    
def colormap_to_colorscale(cmap, n):
    #function that transforms a matplotlib colormap to a Plotly colorscale
    return [ [k*(1/(n-1)), colors.rgb2hex(cmap(k*(1/(n-1))))] for k in range(n)]

def colorscale_from_list(alist, name): 
    # Defines a colormap, and the corresponding Plotly colorscale from the list alist
    # alist=the list of basic colors
    # name is the name of the corresponding matplotlib colormap
    
    cmap = LinearSegmentedColormap.from_list(name, alist)
#    display_cmap(cmap)
    colorscale=colormap_to_colorscale(cmap, 11)
    return cmap, colorscale

def normalize(x,a,b): #maps  the interval [a,b]  to [0,1]
    if a>=b:
        raise ValueError('(a,b) is not an interval')
    return float(x-a)/(b-a)

def asymmetric_colorscale(data,  div_cmap, ref_point=0.0, step=0.05):
    #data: data can be a DataFrame, list of equal length lists, np.array, np.ma.array
    #div_cmap is the symmetric diverging matplotlib or custom colormap
    #ref_point:  reference point
    #step:  is step size for t in [0,1] to evaluate the colormap at t
   
    if isinstance(data, pd.DataFrame):
        D = data.values
    elif isinstance(data, np.ma.core.MaskedArray):
        D=np.ma.copy(data)
    else:    
        D=np.asarray(data, dtype=np.float) 
    
    dmin=np.nanmin(D)
    dmax=np.nanmax(D)
    if not (dmin < ref_point < dmax):
        raise ValueError('data are not appropriate for a diverging colormap')
        
    if dmax+dmin > 2.0*ref_point:
        left=2*ref_point-dmax
        right=dmax
        
        s=normalize(dmin, left,right)
        refp_norm=normalize(ref_point, left, right)# normalize reference point
        
        T=np.arange(refp_norm, s, -step).tolist()+[s]
        T=T[::-1]+np.arange(refp_norm+step, 1, step).tolist()
        
        
    else: 
        left=dmin
        right=2*ref_point-dmin
        
        s=normalize(dmax, left,right) 
        refp_norm=normalize(ref_point, left, right)
        
        T=np.arange(refp_norm, 0, -step).tolist()+[0]
        T=T[::-1]+np.arange(refp_norm+step, s, step).tolist()+[s]
        
    L=len(T)
    T_norm=[normalize(T[k],T[0],T[-1]) for k in range(L)] #normalize T values  
    return [[T_norm[k], colors.rgb2hex(div_cmap(T[k]))] for k in range(L)]

#---------------------------
# end colorbar functions
#---------------------------

def plotClusterSpecificity(experiment, corr_list, threshold, relative=False):
    
    corr_path = os.path.join(data_dir, 'cluster_evaluation', 'k_correlations')
    n_corr = len(corr_list)    
    
    #Create dataframes for plot
    subplt_titls = ()
    titles = []
    for corr in corr_list:
        title = corr+' cluster assignment'
        titles.append((title, None))    
    for t in titles:
        subplt_titls += t
    
    #Initialise plot
    fig = tools.make_subplots(rows=n_corr, cols=2, shared_xaxes=False, print_grid=False, 
                              subplot_titles=subplt_titls)
    #Create colour scale
    colours = cl.scales['12']['qual']['Paired']
    colours = [colours[i] for i in [1,3,5,7,9,11,0,2,4,6,8,10]] + cl.scales['12']['qual']['Set3']
    slatered=['#232c2e', '#ffffff','#c34513']
    label_cmap, label_cs = colorscale_from_list(slatered, 'label_cmap') 
    
    i = 1
    for corr in corr_list:
        
        df = pd.read_csv(os.path.join(corr_path, corr+'_corr.csv'), header=[0]).drop_duplicates(
                    subset=['k','experiment'], keep='last').set_index('k', drop=True)
        df = df.where(df.cluster_size>threshold, np.nan) #exclude k with low membership from viz
        lklhd = df[df.experiment == experiment+'BEST1'].drop(['experiment','cluster_size'], axis=1)
        
        if relative == False:
            pass
        else:
            lklhd = lklhd.divide(relative[i-1], axis=1)
        try:
            ref = 1/sum(relative[i-1])
            weight = 'weighted'
        except:
            ref = 1/lklhd.shape[1]
            weight = ''

        #Create colorscales
        colorscl= asymmetric_colorscale(lklhd, label_cmap, ref_point=ref)

        #Create traces
        heatmap = go.Heatmap(z = lklhd.T.values, x = lklhd.index, y = lklhd.columns, name = corr,
                             colorscale=colorscl, colorbar=dict(title=weight+' likelihood', len=0.9/n_corr, 
                                                                y= 1-i/n_corr+0.05/i, yanchor='bottom'))
        linegraph = {'data': list()}
        for col in range(len(lklhd.columns)):
            linegraph['data'].append({'line': {'color': colours[col], 'width':1.5},
                   'name': lklhd.columns[col],
                   'type': u'scatter',
                   'x': lklhd.index.values,
                   'y':lklhd.iloc[:,col]})

        fig.append_trace(heatmap, i, 1)
        for l in linegraph['data']:
            fig.append_trace(l, i, 2)       
        fig['layout']['yaxis'+str(i*2)].update(title=weight+' likelihood of assignment')        
        i += 1

    #Update layout
    fig['layout'].update(title='Temporal specificity of ' + experiment, height=n_corr*400, hovermode = "closest", showlegend=False) 

    po.plot(fig, filename=clustering_evaluation_dir+'/cluster_specificity_'+experiment+'_'+'_'.join(corr_list)+'.html')


def plotClusterMetrics(metrics_dict, title, metric=None, make_area_plot=False, ylog=False):

    #format plot attributes
    if make_area_plot == True:
        fillme = 'tozeroy'
    else:
        fillme = None
    
    colours = cl.scales['8']['qual']['Dark2']
    
    #generate plot data
    traces = []
    s = 0
    for k,v in metrics_dict.items():
        for i,j in v.items():
            if metric is None:
                grouped=i
                pass
            elif i in metric:
                grouped=None
                pass
            else:
                continue
            x = j.index
            y = j
            traces.append(dict(
                x=x,
                y=y,
                name=k+' | '+i,
                legendgroup=grouped,
                mode='lines',
                marker=dict(size=3),
                line=dict(color=colours[s]),
                fill=fillme,
                connectgaps=True
            ))
        s += 1

    #set layout
    if ylog == True:
        yax = dict(title = 'metric (log scale)' , type='log')
    else:
        yax = dict(title = 'metric')
    layout = go.Layout(
            title= 'Comparison of '+title+' for different model runs',
            margin=go.Margin(t=50,r=50,b=50,l=50, pad=10),
            height= 300+len(traces)*15,
            xaxis=dict(title = 'n clusters'),
            yaxis=yax,
            hovermode = "closest"
            )

    fig = {'data':traces, 'layout':layout }
    return po.plot(fig, filename=clustering_evaluation_dir+'/cluster_metrics_'+title.replace(" ", "_")+'.html')
    
def subplotClusterMetrics(metrics_dict, title, metric=None, make_area_plot=False, ylog=False):

    #format plot attributes
    if make_area_plot == True:
        fillme = 'tozeroy'
    else:
        fillme = None
    
    colours = cl.scales['8']['qual']['Dark2']
    
    fig = tools.make_subplots(rows=2, cols=2, shared_xaxes=False, print_grid=False,
                              subplot_titles= ['mape', 'mdape', 'mdlq', 'mdsyma'])

    #generate plot data
    s = 0
    for k,v in metrics_dict.items():
        t = 0
        ro = 0
        for i,j in v.items():
            co = t%2 + 1
            ro = int(t/2) + 1
            trace = go.Line(
                x= j.index,
                y= j,
                name=k + '|' + i,
                mode='lines',
                legendgroup=k,
                marker=dict(size=3),
                line=dict(color=colours[s]),
                fill=fillme,
                connectgaps=True                    
                )
            fig.append_trace(trace, ro, co)
            t += 1
        s += 1

    #set layout
    if ylog == True:
        yax = dict(title = 'metric (log scale)' , type='log')
    else:
        yax = dict(title = 'metric')

    fig['layout'].update(
            title= 'Comparison of '+title+' for different model runs',
            margin=go.Margin(t=50,r=50,b=50,l=50, pad=10),
            hovermode = "closest"
            )

    for i in range(1,5):
        fig['layout']['yaxis'+str(i)].update(yax)
        fig['layout']['xaxis'+str(i)].update(title='n clusters')

    return po.plot(fig, filename=clustering_evaluation_dir+'/cluster_metrics_'+title.replace(" ", "_")+'.html')

def plotKDispersion(experiment, k, xlabel, centroids):
    
    colour_in = xlabel[['k','elec_bin']].drop_duplicates().reset_index(drop=True).set_index('k').sort_index()
    rcolour_in = ec.renameBins(colour_in, centroids)
    colours = plotPrettyColours(rcolour_in, 'elec_bin')
    
    kxl = xlabel[xlabel['k'] == k]
    
    std_upper = kxl.iloc[:,0:24].describe().loc['mean'] + kxl.iloc[:,0:24].describe().loc['std']
    std_lower = kxl.iloc[:,0:24].describe().loc['mean'] - kxl.iloc[:,0:24].describe().loc['std']

    trace1 = go.Scatter(
        y=kxl.iloc[:,0:24].describe().loc['50%'],
#        fill='tonexty',
#        fillcolor='rgba'+colours[k][3:].replace(')',', 0.1)'),
        mode='lines',
        name='median demand',
        line=dict(
                color=colours[k],
                width = 3,
                dash = 'dash'),
        hoverinfo='all'
        )

    trace2 = go.Scatter(
        y=kxl.iloc[:,0:24].describe().loc['mean'],
        fill='tonexty',
        fillcolor='rgba'+colours[k][3:].replace(')',', 0.1)'),
        mode='lines',
        name='mean demand',
        line=dict(
            color=colours[k],
            width = 3),
        hoverinfo='all'
            )

    trace3 = go.Scatter(
        y=std_upper,
        mode='lines',
        name='stdev',
        fill='tonexty',
        fillcolor='rgba'+colours[k][3:].replace(')',', 0.1)'),
        legendgroup = 'stdev',
        line=dict(
            color=colours[k],
            width = 1),
        hoverinfo='all'
            )
    
    trace4 = go.Scatter(
        showlegend=False,
        fill='tonexty',
        fillcolor='rgba'+colours[k][3:].replace(')',', 0.1)'),
        y=std_lower.where(std_lower>0, 0),
#        name='stdev_lower',
        mode='lines',
        legendgroup = 'stdev',
        line=dict(
            color=colours[k],
            width = 1),
        hoverinfo='all'
            )

    trace5 = go.Scatter(
        y=kxl.iloc[:,0:24].describe().loc['max'],
        mode='markers',
        name='max',
        marker=dict(
            color=colours[k],
            size = 5),
        hoverinfo='all'
            )

    layout = {'title':'Hourly dispersion of all load profiles assigned to cluster '+str(k),
              'showlegend':True,             
              'xaxis':dict(title='time of day (hour)',
                           rangemode='tozero'),
              'yaxis':dict(title='hourly demand (A)', 
                         rangemode='tozero')
          }

    traces = [trace1, trace2, trace3, trace4, trace5]
    fig = go.Figure(data=traces, layout=layout)
    
    del kxl
    
    return po.plot(fig, filename=cluster_analysis_dir+'/'+experiment+'_cluster_dispersion_k'+str(k)+'.html')

def plotClusterConsistency(xlabel, y_axis='daily_demand', colour_var='stdev'):
    """ 
    This function creates a scatter plot of mean household entropy vs mean profile standard deviation for all clusters. The marker size provides an indication of the number of households that use a particular cluster most frequently.
    """   
    
    cv_out = ec.clusterReliability(xlabel)
    hovertext = list()
    for k, c in cv_out.iterrows():
        hovertext.append('cluster {}<br />max count for {} households<br />{:.0f}A mean daily demand<br />mean standard deviation: {:.2f} '.format(k, int(c['hh_count']), c['daily_demand'], c['stdev']))
    trace1 = go.Scatter(
            x=cv_out['entropy'],
            y=cv_out[y_axis],
            mode='markers',
            marker=dict(
                size=cv_out.hh_count**0.5 + 5,
                color = cv_out[colour_var], #set color equal to a variable
                colorscale='Blackbody',
                line = dict(width = 0.75,
                            color = 'rgba(68, 68, 68, 1)'),
                showscale=True,
                colorbar=dict(title=colour_var)),
            text=hovertext,
            hoverinfo='text'
        )
            
    layout = go.Layout(
            title= 'Cluster consistency: entropy vs ' +y_axis.replace('_',' '),
            xaxis = dict(title='mean cluster entropy of households assigned to cluster (based on frequency of use)', rangemode='tozero'),
            yaxis = dict(title='mean '+y_axis.replace('_',' ')+' of profiles assigned to cluster'),
            hovermode= 'closest'
            )
            
    fig= go.Figure(data=[trace1], layout=layout)
    
    return po.plot(fig, filename=cluster_analysis_dir+'/cluster_consistency_Y_'+y_axis+'.html')

def plotHouseholdVolatility(xlabel, colorvar, centroids, legendgroups = None):
    """ 
    This function creates a scatter plot of household entropy vs mean daily demand for all households.
    colorvar displays an additional dimensino and can be one of Year, Municipality, k, stdev
    """
   
    hh_out = ec.householdReliability(xlabel)
    trace_vars = hh_out[colorvar].unique()
    trace_vars.sort()   

    if colorvar == 'k':
        colour_in = xlabel[['k','elec_bin']].drop_duplicates().reset_index(
                drop=True).set_index('k').sort_index()
        rcolour_in = ec.renameBins(colour_in, centroids)
        colour_gradient = plotPrettyColours(rcolour_in, 'elec_bin')
        c1 = dict(zip(colour_gradient.keys(), [colour_gradient[i].replace(')',', 0.8)') for i in colour_gradient.keys()]))
        colour_dict = dict(zip(c1.keys(), [c1[x].replace('rgb','rgba') for x in c1.keys()]))
    else:
        colour_gradient = colormap_to_colorscale(cm.nipy_spectral, len(hh_out[colorvar].unique())+1)
        colour_dict = dict(zip(trace_vars, ['rgba'+str(colors.to_rgba(colour_gradient[x][1], 0.8))  for x in range(len(colour_gradient))]))
    
    traces = []
    i = 0
    for var in trace_vars:
        sub_hh_out = hh_out.loc[hh_out[colorvar]==var]
        if legendgroups != None:
            legendgroups = sub_hh_out['elec_bin'].unique()[0]
        x = sub_hh_out['entropy']
        y = sub_hh_out['daily_demand']
        hovertext = list()
        for k, h in sub_hh_out.iterrows():
            hovertext.append('most used cluster: {}<br />frequency: {} days<br />{}'.format(int(h['k']), int(h['k_count']), h['Year']))

        traces.append(dict(
            x=x,
            y=y,
            name=str(var),
            mode='markers',
            marker=dict(
                size=sub_hh_out['k_count']**0.4 + 1,
                color = colour_dict[var], 
                line = dict(width = 0.5,
                            color = 'rgba(48, 48, 48, 1)')),
            legendgroup=legendgroups,
            text=hovertext,
            hoverinfo='text'
        ))
        i += 1
    
    layout = go.Layout(
            title= 'Household Characterisation: entropy vs daily energy demand (A) - coloured by '+colorvar,
            xaxis = dict(title='household entropy', rangemode='tozero'),
            yaxis = dict(title='mean daily demand (A)', rangemode='tozero'),
            hovermode= 'closest'
            )
            
    fig= go.Figure(data=traces, layout=layout)
    
    del hh_out, traces#trace1
       
    return po.plot(fig, filename=cluster_analysis_dir+'/household_volatility_'+colorvar+'.html')

def plotHouseholdReliability(F, month, daytype):
        
    Xsub = F[(F['season']==month)&(F['daytype']==daytype)].groupby(['elec_bin','k','ProfileID'])['DD'].count().reset_index()
    
    colour_data = Xsub.drop_duplicates(subset=['k','elec_bin']).set_index('k',drop=True)    
    colours =  plotPrettyColours(colour_data, 'elec_bin') 
    
    sub_plots = Xsub.elec_bin.unique()
    fig = tools.make_subplots(rows=len(sub_plots), cols=1, shared_xaxes=False, 
                              subplot_titles=sub_plots, print_grid=False)  
    
    i = 1
    for s in sub_plots:
        for j in Xsub.loc[Xsub.elec_bin==s, 'k'].unique():
            fig.append_trace({
                'x' : ['p'+ str(p) for p in Xsub.loc[(Xsub.elec_bin==s) & (Xsub.k==j), 'ProfileID'].values],
                'y' : Xsub.loc[(Xsub.elec_bin==s) & (Xsub.k==j),'DD'],
                'name' : j,
                'legendgroup' : s,
                'marker' : dict(color=colours[j]),
                'type' : 'bar'}, i, 1)
        i += 1

    fig['layout'].update(barmode='stack',
                         height=len(sub_plots)*300,
                         title=month + ' ' + daytype + ' cluster count for households')
    
    del Xsub

    return po.iplot(fig)

#plot = cv_out.plot.scatter(x='entropy', y='stdev', s=np.sqrt(cv_out.hh_count)*5, c='daily_demand', colormap='nipy_spectral')
    