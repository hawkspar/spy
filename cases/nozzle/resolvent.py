# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from setup import *
from spyp import SPYP # Must be after setup
from mpi4py.MPI import COMM_WORLD as comm
import numpy as np

p0=comm.rank==0
ms=np.arange(1,6)
Sts=np.linspace(0,1,20)

for m in ms:
	spyp=SPYP(params,datapath,Ref,nutf,direction_map,0,m)
	boundaryConditionsPerturbations(spyp,m)
	# For efficiency, matrices assembled once per Sts
	spyp.assembleJNMatrices()
	spyp.assembleMRMatrices()
	# Resolvent analysis
	spyp.resolvent(1,Sts)