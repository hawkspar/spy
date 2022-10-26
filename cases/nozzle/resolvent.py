# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import numpy as np
from setup import *
from spyp import SPYP # Must be after setup
from mpi4py.MPI import COMM_WORLD as comm

ms=[1,2]
Sts=np.linspace(0,1,11)
Ss=[0,.1]
Re=400000

spyp=SPYP(params,datapath,direction_map)#,forcingIndicator)
for S in Ss:
	for m in ms:
		Ref(spyp,1000)
		nutf(spyp,Re,S)
		spyp.loadBaseflow(Re,S) # Don't load pressure
		boundaryConditionsPerturbations(spyp,m)
		spyp.stabilise(m)
		# For efficiency, matrices assembled once per Sts
		spyp.assembleJNMatrices(weakBoundaryConditions,m)
		spyp.assembleMRMatrices()
		# Resolvent analysis
		spyp.resolvent(1,Sts,Re,S,m)