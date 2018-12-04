# South African Residential Energy Consumer Archetypes

```bash
dlr_sarec_archetypes
	|-- img
	|-- log
	|-- results
	|-- src
		|-- data
		|-- evaluation
			|-- benchmark.py
			|-- eval_classification.py
			|-- eval_cluster_plot.py
			|-- eval_clusters.py
		|-- params
			|-- *.txt
		|-- pipeline
			|-- linux
			|-- windows
			|-- *.py (can probably delete and put in command_line.py)
		|-- cluster_metrics.py
		|-- clusters.py
		|-- command_line.py
		|-- support.py
	|-- LICENSE
	|-- README.md
	|-- setup.py
``` 
	
## About this package
This package constructs *S*outh *A*frican *r*esidential *e*nergy *c*onsumer archetypes from South Africa's Domestic Load Research database.

## Setup instructions
Ensure that python 3 is installed on your computer. A simple way of getting it is to install it with [Anaconda](https://conda.io/docs/user-guide/install/index.html). 

1. Clone this repository from github.
2. Navigate to the root directory (`dlr_sarec_archetypes`) and run the `setup.py` script
3. You will be asked to confirm the data directories that contain your input data. Paste the full path name when prompted. You can change this setting at a later stage by modifying the file `src/data/store_path.txt` .

## Overview of Process

1. Generate input data from DLR load profiles (24 hour profiles for households surveyed in year range)
2. 
3. 

### Command-line Interface

Once the package has been installed, the clustering process can be run from the command line using the following commands:

```


```

### Clustering Parameters

Clustering paramters are passed to functions in the package `dlr_clusters`. Algorithms can be constructed with the following parameters specified in a `params\*.txt` file.

```
algorithm: {kmeans, som} #choose clustering algorithm  
start: year in [1994,2014] #select survey group start year  
end: year in [1994,2014] #select survey group end year  
drop_0: {True, False} #drop 0-value profiles  
preprocessing: {None, unit_norm, demin, zero-one, sa_norm} #select normalisation algorithm  
bin_X: {integral, amd} #prebin by integral kmeans or average monthly demand  
range_n_dim: {None, range(start, end, step)} #specify som dimensions  
transform: {None, kmeans} #select kmeans for som+kmeans  
range_n_clusters: {None, range(start, end, step)} #specify range of kmeans clusters  

```

### Evaluation

`dlr_cluster_metrics` implements the Davies Bouldin Index, Silhouette Index and Mean Index Adequacy and an 'all_scores' index calculated from the product of the three scores. The functions are used to compute quantitative cluster metrics when running clustering algorithms.

This project implements a further qualitative evaluation process. Functions for this process are contained in the package `evaluation.eval_clusters` and can be visualised with `evaluation.eval_cluster_plot`.

### Classification

** requires [dlr_data_processing]()

arff files can be generated to pass as input into [WEKA's]() classification algorithms.

## Publications
The methods used in this work have been published in:
[]()
