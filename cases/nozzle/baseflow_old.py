# coding: utf-8
"""
Created on Wed Sept  28 09:28:00 2022

@author: hawkspar
"""
from setup import *
from spyb import SPYB

spyb=SPYB(params,datapath,"perturbations",direction_map)
boundaryConditionsBaseflow(spyb,S)
# Shorthands
d=dist(spyb)
Ref(spyb,Re)
spyb.stabilise(0)
spyb.loadBaseflow(nut,nut,S)
spyb.baseflow(Re,nut,S,d,(0,0),baseflowInit=baseflowInit)