# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
#import os, re
from setup import *
from spyb import SPYB # Must be after setup
#from spy  import meshConvert
#from mpi4py.MPI import COMM_WORLD as comm

#meshConvert("/home/shared/cases/garnaud/garnaud",'triangle')
#def nutf(spy,_): spy.nut=0
spyb=SPYB(params,datapath,lambda _: Re, nutf,direction_map)
# Baseflow calculation
boundaryConditionsBaseflow(spyb)
spyb.baseflow(Re,S,True)#,baseflowInit=baseflowInit)

"""if comm.rank==0:
	file_names = [f for f in os.listdir(spyb.dat_real_path)]
	for file_name in file_names:
		match = re.search(r'_Re=(\d*)',file_name)
		if 1000 != int(match.group(1)): os.remove(spyb.dat_real_path+file_name)
comm.barrier()

spyb.datToNpyAll()"""