# coding: utf-8
"""
Created on Wed Sept  28 09:28:00 2022

@author: hawkspar
"""
from setup import *
from spyb import SPYB
from mpi4py.MPI import COMM_WORLD as comm

spyb=SPYB(params,datapath,Ref,nutf,direction_map,True)
spyb.loadBaseflow(S,Re,True)
boundaryConditionsBaseflow(spyb)
if comm.rank==0: print("BCs fixed",flush=True)
spyb.smoothenBaseflow()
spyb.saveBaseflow(f"_S={S:00.3f}_Re={Re:d}")