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
      install_requires=['pandas','numpy','feather-format','plotly','colorlover','scikit-learn',
                        'peakutils','pathlib','somoclu'],
      include_package_data=True,
      packages=find_packages(),
      py_modules = ['delarchetypes.command_line','delarchetypes.cluster.cluster','delarchetypes.cluster.results',
      'delarchetypes.cluster.qualeval','delarchetypes.classify.features'],
      entry_points = {
			'console_scripts': ['delarch_cluster=delarchetypes.command_line:clustersGen',
                       'delarch_cluster_eval=delarchetypes.command_line:clustersEval',
                       'delarch_prep_classify=delarchetypes.command_line:prepClassify'],
                       }
      )
