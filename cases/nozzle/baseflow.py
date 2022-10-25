# coding: utf-8
"""
Created on Wed Sept  28 09:28:00 2022

@author: hawkspar
"""
from setup import *
from spyb import SPYB

def weakBoundaryConditions(spy,u,p,m=0): return 0

spyb=SPYB(params,datapath,direction_map)
boundaryConditionsBaseflow(spyb)
spyb.loadBaseflow(1000,S,True)
Ref(spyb,1000)
nutf(spyb,1000,S)
spyb.sanityCheck("_load")
spyb.smoothen()
spyb.sanityCheckU("_smooth")
spyb.stabilise(0)
spyb.baseflow2(Re,S,weakBoundaryConditions)
for Re in [10000, 100000, 400000]:
	nutf(spyb,Re,S)
	spyb.baseflow(Re,S,weakBoundaryConditions)
spyb.baseflow(400000,.1,weakBoundaryConditions)