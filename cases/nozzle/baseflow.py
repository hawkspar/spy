# coding: utf-8
"""
Created on Wed Sept  28 09:28:00 2022

@author: hawkspar
"""
from setup import *
from spyb import SPYB

def weakBoundaryConditions(spy,u,p,m=0): return 0

spyb=SPYB(params,datapath,direction_map)
"""boundaryConditionsBaseflow(spyb,0)
spyb.loadBaseflow(Re,S,True)
spyb.smoothenBaseflow(boundaryConditionsU(spyb),lambda spy,u,v: 0)
Ref(spyb,1000)
nutf(spyb,1000,S)
spyb.stabilise(0)
spyb.baseflow(Re,S,weakBoundaryConditions,baseflowInit=baseflowInit)
for Re in [10000, 100000, 400000]:
	#Ref(spyb,Re)
	nutf(spyb,Re,S)
	spyb.baseflow(Re,S,weakBoundaryConditions)"""
spyb.loadBaseflow(400000,0,True)
Ref(spyb,1000)
nutf(spyb,400000,.1)
spyb.stabilise(0)
for S in [1e-3,1e-2]:
	boundaryConditionsBaseflow(spyb,S)
	spyb.baseflow(400000,S,weakBoundaryConditions,save=False)
boundaryConditionsBaseflow(spyb,.1)
spyb.baseflow(400000,.1,weakBoundaryConditions)