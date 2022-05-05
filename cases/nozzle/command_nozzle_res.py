# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from nozzle_setup import *
from spyp import SPYP # Must be after setup

spyp=SPYP(params,datapath,Ref,nutf,boundaryConditionsPerturbations,0,0)
# For efficiency, matrix is assembled only once
spyp.assembleMatrices()
# Resolvent analysis
spyp.resolvent(1,[.2])