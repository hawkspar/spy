# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from setup import *
from spyp import SPYP # Must be after setup
from spy import meshConvert
from mpi4py.MPI import COMM_WORLD as comm

if comm.rank==0:
	meshConvert("nozzle_2D_coarser","nozzle_coarser","quad")
comm.barrier()

"""ms=range(6)
Sts=np.hstack((np.linspace(1,.2,20,endpoint=False),np.linspace(.2,.1,5,endpoint=False),np.linspace(.1,.01,10)))
Ss=[0,1]"""
ms=[3]
Sts=[.01]
Ss=[1]
spyp=SPYP(params,datapath,direction_map)
for str in ["forcing","response"]:
	for m in ms:
		for St in Sts:
			for S in Ss:
				#spyp.visualiseCurls(str,1000,400000,S,m,St,.5)
				spyp.visualise3dModes(str,1000,400000,S,m,St)
				#spyp.visualiseStreaks(str,1000,400000,S,m,St,5)