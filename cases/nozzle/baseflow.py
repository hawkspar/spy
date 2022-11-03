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
nutf(spyb,400000,0)
spyb.stabilise(0)
spyb.baseflow(1000,400000,0,baseflowInit=baseflowInit)
# Now to the swirling flow
nutf(spyb,400000,.1)
for S in [1e-2,.1]:
	boundaryConditionsBaseflow(spyb,S)
	spyb.baseflow(1000,400000,S)