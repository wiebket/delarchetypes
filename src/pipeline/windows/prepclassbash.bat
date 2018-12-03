#!/bin/bash

set root=C:\Users\CKAN\Anaconda3
call %root%\Scripts\activate.bat %root%

clear
echo "Ready to roll."

python prepClassify.py exp5_kmeans_unit_norm features1
python prepClassify.py exp5_kmeans_unit_norm features2 --skip_cat adults children part_time pension unemployed
python prepClassify.py exp5_kmeans_unit_norm features3 --skip_cat adults children part_time pension unemployed cb_size floor_area monthly_income years_electrified