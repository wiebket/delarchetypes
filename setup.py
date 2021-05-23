#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 14:17:10 2018

@author: SaintlyVi
"""
from setuptools import setup, find_packages
import os
from pathlib import Path

usr_dir = os.path.join(Path.home(), 'del_data','usr')
os.makedirs(usr_dir, exist_ok=True)

setup(
      name = 'delarchetypes',
      version= 0.1,
      decsription = 'creates customer archetypes',
      long_description = '',
      keywords='domestic load research south africa data processing',
      url='https://github.com/wiebket/delarchetypes ',
      author='Wiebke Toussaint',
      author_email='wiebke.toussaint@gmail.com',
      license='CC-BY-NC',
      install_requires=['pandas','numpy','pyodbc','feather-format','plotly', 
                        'pathlib','pyshp','shapely','somoclu'],
      include_package_data=True,
      packages=find_packages(),
      py_modules = ['delarchetypes.command_line', 'delarchetypes.support', 'delarchetypes.cluster.cluster'],
      data_files=[(os.path.join(usr_dir,'specs'), [os.path.join('delarchetypes','experiment','parameters', f) for f in [files for root, dirs, files in os.walk(os.path.join('delarchetypes','experiment','parameters'))][0]])],
      entry_points = {
			'console_scripts': ['delarchetypes_clusters=delarchetypes.command_line:clustersGen',
                       'delarchetypes_clustersEval=delarchetypes.command_line:clustersEval',
                       'delarchetypes_classify=delarchetypes.command_line:prepClassify'],
                       }
      )
