# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import os, re
from setup import *
from spyb import SPYB # Must be after setup
from spy  import meshConvert
from mpi4py.MPI import COMM_WORLD as comm

#meshConvert("/home/shared/cases/garnaud/garnaud",'triangle')

spyb=SPYB(params,datapath,lambda _: 10,nutf,direction_map,InletAzimuthalVelocity)
# Baseflow calculation
boundaryConditionsBaseflow(spyb)
for Re in [10, 50, 250, 1000]:
#for Re in [1000]:
	spyb.Re=Re
	spyb.baseflow(Re,0,True,baseflowInit=baseflowInit)

if comm.rank==0:
	file_names = [f for f in os.listdir(spyb.dat_real_path)]
	for file_name in file_names:
		match = re.search(r'_Re=(\d*)',file_name)
		if 1000 != int(match.group(1)): os.remove(spyb.dat_real_path+file_name)
comm.barrier()

spyb.datToNpyAll()