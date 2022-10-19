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
spyb=SPYB(params,datapath,direction_map)
spyb.loadBaseflow(S,Re,True)
Ref(spyb)
spyb.nut=0
#nutf(spyb,S,Re)
# Baseflow calculation
spyb.stabilise(0)
boundaryConditionsBaseflow(spyb)
"""U,P=spyb.Q.split()
def P_init(x):
	x,r=x[0],x[1]
	p=0*r
	p[r<1]=-x[r<1]
	return p
P.interpolate(P_init)"""
spyb.baseflow(Re,S,weakBoundaryConditions)#,baseflowInit=baseflowInit)

"""if comm.rank==0:
	file_names = [f for f in os.listdir(spyb.dat_real_path)]
	for file_name in file_names:
		match = re.search(r'_Re=(\d*)',file_name)
		if 1000 != int(match.group(1)): os.remove(spyb.dat_real_path+file_name)
comm.barrier()

spyb.datToNpyAll()"""