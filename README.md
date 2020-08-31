# Clustering Pipeline for South African Residential Energy Consumer Archetypes

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
This package is a pipeline constructs South African residential energy consumer archetypes from the South African Domestic Electric Load (DEL) database. It requires access to the [DELMH](https://doi.org/10.25828/56nh-fw77) and [DELMSKV](https://doi.org/10.25828/mf8s-hh79) datasets from the NRS Load Research study. Data access can be requested from Data First at the University of Cape Town.   

The data descriptor for the dataset is available [online](https://doi.org/10.25375/uct.11774691.v1).

Two other packages have been released alongside this package for [retrieving](https://github.com/wiebket/delretrieve) and [processing](https://github.com/wiebket/delprocess) data from the DEL database. 

## Setup instructions
Ensure that python 3 is installed on your computer. A simple way of getting it is to install it with [Anaconda](https://docs.anaconda.com/anaconda/install/). Once python has been installed, the delprocess package can be installed.
	
1. Clone this repository from github.
2. Navigate to the root directory (`delarchetype`) and run `python setup.py install` (run from Anaconda Prompt or other bash with access to python if running on Windows).
3. You will be asked to confirm the data directories that contain your data. Paste the full path name when prompted. You can change this setting at a later stage by modifying the file `your_home_dir/del_data/usr/store_path.txt` .

This package only works if the data structure is _exactly_ like the directory hierarchy in _del_data_ if created with the package `delretrieve`:

```bash
your_home_dir/del_data
    |-- observations
        |-- profiles
            |-- raw
                |-- unit
                    |-- GroupYear
        |-- tables
            |-- ...
    |-- survey_features
    |-- usr
        |-- specs (automatically copied from delprocess/data/specs during setup)
        |-- store_path.txt (generated during setup)
```

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
