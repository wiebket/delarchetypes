#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 14:39:38 2018

@author: saintlyvi
"""

import argparse

from features.feature_extraction import genFProfiles, genArffFile 

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