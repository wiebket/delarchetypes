#!/bin/bash

set root=C:\Users\CKAN\Anaconda3
call %root%\Scripts\activate.bat %root%

clear
echo "Ready to roll."

python clustersEval.py exp5_kmeans_unit_norm features1
python clustersEval.py exp5_kmeans_unit_norm features2
python clustersEval.py exp5_kmeans_unit_norm features3
