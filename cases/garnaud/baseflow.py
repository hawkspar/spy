# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import os, re
from setup import *
from spyb import SPYB # Must be after setup
from mpi4py.MPI import COMM_WORLD as comm

start_Re=10
spyb=SPYB(params,datapath,lambda _: start_Re,nutf,direction_map)
# Baseflow calculation
boundaryConditionsBaseflow(spyb)
spyb.Re=start_Re
spyb.baseflow(start_Re,S,False,baseflowInit=baseflowInit)
for cur_Re in [50, 250, Re]:
	spyb.Re=cur_Re
	spyb.baseflow(cur_Re,S,True)
# Purge Re<1000
if comm.rank==0:
	file_names = [f for f in os.listdir(spyb.u_path)]
	for file_name in file_names:
		match = re.search(r'_Re=(\d*)',file_name)
		if int(match.group(1)) != 1000: os.remove(spyb.u_path+file_name)