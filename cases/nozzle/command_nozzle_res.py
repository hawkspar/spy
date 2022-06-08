# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from nozzle_setup import *
from spyp import SPYP # Must be after setup
from mpi4py.MPI import COMM_WORLD as comm
import time

p0=comm.rank==0
m=0
ts=np.empty(10)
for i in range(10):
	if p0: start = time.perf_counter()
	spyp=SPYP(params,datapath,Ref,nutf,direction_map,0,m)
	boundaryConditionsPerturbations(spyp,m)
	# For efficiency, matrix is assembled only once
	spyp.assembleJNMatrices()
	spyp.assembleMRMatrices()
	# Resolvent analysis
	spyp.resolvent(2,[.6])
	if p0:
		end = time.perf_counter()
		ts[i]=end-start
	print(f"Runtime: {end-start}")
print(f"Average runtime: {np.mean(ts)}")