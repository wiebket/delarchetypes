# Clustering Residential Electricity Consumption Data in South Africa
	
## About
This package is a pipeline that constructs South African residential electricity consumer archetypes from the South African Domestic Electric Load (DEL) database. 

This is the code repository for the following published research:  

1. Toussaint, W. and Moodley, D. 2020. “Clustering Residential Electricity Consumption Data to Create Archetypes that Capture Household Behaviour in South Africa”. South African Computer Journal 32(2), 1–34. [https://doi.org/10.18489/sacj.v32i2.845]
2. Toussaint, W. and Moodley, D. 2020. “Identifying optimal clustering structures for residential energy consumption patterns using competency questions.” In Conference of the South African Institute of Computer Scientists and Information Technologists 2020 (SAICSIT ‘20). Association for Computing Machinery, New York, NY, USA, 66–73. [https://doi.org/10.1145/3410886.3410887]
3. Toussaint, W. and Moodley, D. 2019. “Comparison of Clustering Techniques for Residential Load Profiles in South Africa.” In Proceedings of the South African Forum for AI Research. URL [CEUR-WS.org/Vol-1//Vol-2540/FAIR2019_paper_55.pdf]
4. Toussaint, Wiebke. Evaluation of Clustering Techniques for Generating Household Energy Consumption Patterns in a Developing Country Context. 2019. URL [http://hdl.handle.net/11427/30905] **[Masters thesis with complete overview of experiments]**

### Data & Pre-Processing
This code requires access to the [DELMH](https://doi.org/10.25828/56nh-fw77) and [DELMSKV](https://doi.org/10.25828/mf8s-hh79) datasets from the NRS Load Research study. Data access can be requested from Data First at the University of Cape Town. The data descriptor for the dataset is available [online](https://doi.org/10.25375/uct.11774691.v1).

Two other packages have been released alongside this package for [retrieving](https://github.com/wiebket/delretrieve) and [processing](https://github.com/wiebket/delprocess) data from the DEL database. 

## Setup instructions
Ensure that python 3 is installed on your computer. A simple way of getting it is to install it with [Anaconda](https://docs.anaconda.com/anaconda/install/). Once python has been installed, the delarchetype package can be installed.

0. Requires: [`delprocess`](https://github.com/wiebket/delprocess)	
1. Clone this repository from github.
2. Navigate to the root directory (`delarchetype`) and run `python setup.py install` (run from Anaconda Prompt or other bash with access to python if running on Windows).
3. During the install process a new directory will be created at `your_home_dir/del_data/` if it does not yet exist. This is your default data directory. You can change this setting at a later stage by modifying the file `your_home_dir/del_data/usr/store_path.txt`.
4. This package only works if the data structure is _exactly_ like the directory hierarchy in _del_data_ if created with the package `delretrieve`. 


## Package Usage

### Command-line Interface

Once the package has been installed, the clustering process can be run from the command line using the following commands:

1. `delarch_cluster params -top -skip` (equivalent to delarchetypes.command_line.clustersGen())
2. `delarch_cluster_eval experiment n_best` (equivalent to delarchetypes.command_line.clustersEval())
3. `delarch_prep_classify experiment socios` (equivalent to delarchetypes.command_line.prepClassify())

Consult `delarchetypes.command_line` for a full list and specification of parameters.

### Clustering Parameters

Clustering paramters are passed to functions in `clusters.clusters`. Algorithms can be constructed with the following parameters specified in a file stored at `delarchetypes\delarchetypes\experiment\parameters\*.txt` 

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
Looking under the hood of `delarch_cluster_eval`:

`cluster.metrics` implements the Davies Bouldin Index, Silhouette Index and Mean Index Adequacy and an 'all_scores' index calculated from the product of the three scores. The functions are used to compute quantitative cluster metrics when running clustering algorithms.

This project implements a further qualitative evaluation process. Functions for this process are contained in the package `clusters.qualeval` and can be visualised with `clusters.plot`.

### Classification
Looking under the hood of `delarch_prep_classify`:

This generates arff files to pass as input into [WEKA](https://www.cs.waikato.ac.nz/ml/weka/)'s classification algorithms.
