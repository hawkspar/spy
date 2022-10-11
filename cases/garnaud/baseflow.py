# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import os, re
from setup import *
from spyb import SPYB # Must be after setup
from mpi4py.MPI import COMM_WORLD as comm

spyb=SPYB(params,datapath,lambda _: 10,nutf,direction_map)
# Baseflow calculation
boundaryConditionsBaseflow(spyb)
"""spyb.Re=10
spyb.baseflow(10,0,False,baseflowInit=baseflowInit)
for Re in [50, 250, 1000]:
	spyb.Re=Re
	spyb.baseflow(Re,0,True)
# Purge Re<1000
if comm.rank==0:
	file_names = [f for f in os.listdir(spyb.u_path+"real/")]
	for file_name in file_names:
		match = re.search(r'_Re=(\d*)',file_name)
		if int(match.group(1)) != 1000: os.remove(spyb.u_path+"real/"+file_name)
comm.barrier()"""

spyb.datToNpyAll({"u":True})