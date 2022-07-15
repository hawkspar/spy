# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import numpy as np
from setup import *
from spyp  import SPYP # Must be after setup

ms=np.arange(0,6)
Sts=np.linspace(0,1,20)

for m in ms[1:]:
	if m==1: Sts=Sts[Sts>.1]
	spyp=SPYP(params,datapath,Ref,nutf,direction_map,0,m)
	boundaryConditionsPerturbations(spyp,m)
	# For efficiency, matrices assembled once per Sts
	spyp.assembleJNMatrices(Re)
	spyp.assembleMRMatrices()
	# Resolvent analysis
	spyp.resolvent(1,Sts)