# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from nozzle_setup import *
from spyp import SPYP # Must be after setup

m=0
spyp=SPYP(params,datapath,Ref,nutf,direction_map,0,m)
boundaryConditionsPerturbations(spyp,m)
spyp.sanityCheckBCs()
# For efficiency, matrix is assembled only once
spyp.assembleJNMatrices()
spyp.assembleMRMatrices()
# Resolvent analysis
spyp.resolvent(1,[.6])