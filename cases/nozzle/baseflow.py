# coding: utf-8
"""
Created on Wed Sept  28 09:28:00 2022

@author: hawkspar
"""
from setup import *
from spyb import SPYB

spyb=SPYB(params,datapath,direction_map)
boundaryConditionsBaseflow(spyb,0)
Ref(spyb,1000)
spyb.stabilise(0)
spyb.baseflow(1000,400000,0,dist,baseflowInit=baseflowInit)
for Re in [5000,10000,20000]:
	spyb.baseflow(Re,400000,0,dist)
"""# Now to the swirling flow
nutf(spyb,400000,.1)
for S in [1e-2,.1]:
	boundaryConditionsBaseflow(spyb,S)
	spyb.baseflow(000,400000,S)
spyb.loadBaseflow(1000,400000,.5,True)
# Now to the more swirling flow
nutf(spyb,400000,.5)
for S in [1]:
	boundaryConditionsBaseflow(spyb,S)
	spyb.baseflow(1000,400000,S)"""