# coding: utf-8
"""
Created on Wed Sept  28 09:28:00 2022

@author: hawkspar
"""
from setup import *
from spyb import SPYB
from mpi4py.MPI import COMM_WORLD as comm

spyb=SPYB(params,datapath,direction_map)
spyb.loadBaseflow(S,Re,True)
Ref(spyb)
nutf(spyb,S,Re)
boundaryConditionsBaseflow(spyb)
if comm.rank==0: print("BCs fixed",flush=True)
spyb.smoothenBaseflow(boundaryConditionsU(spyb),lambda spy,u,v: 0) # No additionnal constraints on u during smoothing, no constraint ever for p or nut
spyb.stabilise(0)
spyb.baseflow(Re,S,lambda spy,u,p: weakBoundaryConditions(spy,u,p,0),baseflowInit=baseflowInit)