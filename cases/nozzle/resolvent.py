# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import numpy as np
from setup import *
from spyp import SPYP # Must be after setup
from mpi4py.MPI import COMM_WORLD as comm

#ms=range(3)
ms=[0]
#Sts=np.linspace(0,1,11)
Sts=[.9]

for m in ms:
	spyp=SPYP(params,datapath,Ref,Re,nutf,direction_map,S,m)#,forcingIndicator)
	spyp.loadBaseflow(S,Re) # Don't load pressure
	boundaryConditionsPerturbations(spyp,m)
	spyp.stabilise(m)
	# For efficiency, matrices assembled once per Sts
	spyp.assembleJNMatrices(weakBoundaryConditions)
	spyp.assembleMRMatrices()
	# Resolvent analysis
	spyp.resolvent(1,Sts)