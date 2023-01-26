# coding: utf-8
"""
Created on Wed Sept  28 09:28:00 2022

@author: hawkspar
"""
from setup import *
from spyb import SPYB
from spy import loadStuff

spyb=SPYB(params,datapath,"perturbations",direction_map)
boundaryConditionsBaseflow(spyb,0)
# Shorthands
d=dist(spyb)
Ref(spyb,Re)
spyb.stabilise(0)
loadStuff(spyb.nut_path,['S','nut','Re'],[0,nut,nut],spyb.Nu)
spyb.baseflow(Re,nut,0,d,baseflowInit=baseflowInit)
for S in np.linspace(.1,1,9):
	spyb.dofs = np.empty(0,dtype=np.int32)
	spyb.bcs=[]
	boundaryConditionsBaseflow(spyb,S)
	loadStuff(spyb.nut_path,['S','nut','Re'],[S,nut,nut],spyb.Nu)
	spyb.baseflow(Re,nut,S,d)