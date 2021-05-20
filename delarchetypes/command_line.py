#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday 3 December 2018

@author: Wiebke Toussaint
"""

import argparse
import os
import pandas as pd
import time
import sys

from delprocess.loadprofiles import genX
from .cluster.cluster import som, kmeans
from .cluster.results import getLabels, realCentroids 
from .cluster.qualeval import consumptionError, peakCoincidence, saveCorr
from .classify.features import genFProfiles, genArffFile

from .support import experiment_dir, writeLog, log_dir

def clustersGen():
	# Set up argument parser to run from terminal
	parser = argparse.ArgumentParser(description='Cluster DLR timeseries data.')
	parser.add_argument('params', type=str, help='Parameter file with clustering specifications')
	parser.add_argument('-top', type=int, help='Save labels for top n results')
	parser.add_argument('-skip', type=int, help='Skip runs from top in parameter file')
	args = parser.parse_args()

	param_dir = os.path.join(experiment_dir, 'parameters')
	param_path = os.path.join(param_dir, args.params + '.txt')
	header = open(param_path,'r')
	param = list()
	for line in header:
		if line.strip() != '':                # ignore blank lines
		    param.append(eval(line))

	if args.skip is None:
		skip_experiments = 0
	else:
		skip_experiments = args.skip

	for i in range(skip_experiments+1, len(param)): #skip first line with header info
		# Extract all parameter values
		algorithm = param[i][0]
		start = param[i][1]
		end = param[i][2]
		drop = param[i][3]
		preprocessing = param[i][4]
		bin_X = param[i][5]
		range_n_dim = param[i][6]
		transform = param[i][7]
		range_n_clusters = param[i][8]
		
		print(param[i])
		
		tic = time.time()
		
		X = genX([start, end], drop_0=drop) # TODO This line might cause trouble reading X... change drop_0 dtype from boolean if that is the case

		if algorithm == 'som':
		    stats, centroids, cluster_lbls = som(X, range_n_dim, args.top, preprocessing, bin_X, transform, args.params,
		                                         n_clusters=range_n_clusters)  
		if algorithm == 'kmeans':
		    stats, centroids, cluster_lbls = kmeans(X, range_n_clusters, args.top, preprocessing, bin_X, args.params)
	#    if args.top:
	#        saveLabels(X, cluster_lbls, stats, args.top)
		    
		toc = time.time()
		
		log_line = param[i]
		logs = pd.DataFrame([[args.params, (toc-tic)/60] + list(log_line)], columns = [
                'experiment','runtime','algorithm', 'start', 'end', 'drop_0', 
                'preprocessing', 'bin_X', 'range_n_dim', 'transform','range_n_clusters'])
		writeLog(logs, os.path.join(log_dir,'log_runClusters'))

	return print('\n>>>Cluster generation complete<<<')

def clustersEval():
	# Set up argument parser to run from terminal
	parser = argparse.ArgumentParser(description='Evaluate DLR timeseries clusters.')
	parser.add_argument('experiment', type=str, help='Experiment_algorithm_preprocessing')
	parser.add_argument('n_best', type=int, help='n_best run of experiment')
	args = parser.parse_args()


	xl = getLabels(args.experiment, args.n_best)
	centroids = realCentroids(args.experiment, xl, args.n_best)
	consumptionError(xl, centroids, compare='total')
	consumptionError(xl, centroids, compare='peak')
	peak_eval = peakCoincidence(xl, centroids)
	saveCorr(xl, args.experiment)
	
	return print("\n>>>Cluster evaluation complete.<<<")


def prepClassify():
    
    # Set up argument parser to run from terminal
    parser = argparse.ArgumentParser(description='Evaluate DLR timeseries clusters.')
    parser.add_argument('experiment', type=str, help='Experiment_algorithm_preprocessing')
    parser.add_argument('socios', type=str, help='Specification of socio_demographic features')
    parser.add_argument('--keep_years', action='store_true', help='Specify if k_count should be by year.')
    parser.add_argument('--filter_features', default=None, help='Filter features by values. Specify dict as string.')
    parser.add_argument('--skip_cat', default=None, nargs='*', help='Specify numeric features')
    parser.add_argument('--weighted', action='store_false', help='Specify if features should be weighted')
    parser.add_argument('--n_best', type=int, default=1, help='n_best run of experiment')
    args = parser.parse_args()
    
    F = genFProfiles(args.experiment, args.socios, args.n_best, args.keep_years, savefig=False)
    genArffFile(args.experiment, args.socios, args.filter_features, args.skip_cat, args.weighted, args.n_best)
    
    return print("\n>>>Feature input prepared for classification.<<<")