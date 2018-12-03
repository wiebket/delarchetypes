#!/bin/bash

set root=C:\Users\CKAN\Anaconda3
call %root%\Scripts\activate.bat %root%

clear
echo "Ready to roll."

python clustersGen.py exp5_kmeans -top 5
#python clustersGen.py exp5_som+kmeans -top 5
