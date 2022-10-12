# coding: utf-8
"""
Created on Wed Sept  28 09:28:00 2022

@author: hawkspar
"""
from setup import *
from spyb import SPYB
from mpi4py.MPI import COMM_WORLD as comm

spyb=SPYB(params,datapath,Ref,nutf,direction_map)
spyb.loadBaseflow(S,Re,True)
spyb.sanityCheck()
boundaryConditionsBaseflow(spyb)
if comm.rank==0: print("BCs fixed",flush=True)
spyb.smoothenBaseflow(lambda spy,u,v,m: 0,weakBoundaryConditionsPressure)
spyb.sanityCheck("_smooth")
#spyb.stabilise(0)
spyb.baseflow(Re,S,weak_bcs=lambda spy,u,p: weakBoundaryConditions(spy,u,p,0))
#spyb.saveBaseflow(f"_S={S:00.3f}_Re={Re:d}")