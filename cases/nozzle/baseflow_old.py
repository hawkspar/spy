# coding: utf-8
"""
Created on Wed Sept  28 09:28:00 2022

@author: hawkspar
"""
from setup import *
from spyb import SPYB
from spy import loadStuff

spyb=SPYB(params,datapath,"perturbations",direction_map)
u_inlet_th,class_th=boundaryConditionsBaseflow(spyb,0)
# Shorthands
d=dist(spyb)
Ref(spyb,Re)
spyb.stabilise(0)
loadStuff(spyb.nut_path,['S','nut','Re'],[0,nut,nut],spyb.Nu)
spyb.baseflow(Re,nut,0,d,baseflowInit=baseflowInit)
#spyb.loadBaseflow(Re,nut,0,True)
for S in np.linspace(1e-2,1,6):
	class_th.S=S
	u_inlet_th.interpolate(class_th)
	loadStuff(spyb.nut_path,['S','nut','Re'],[S,nut,nut],spyb.Nu)
	spyb.baseflow(Re,nut,S,d)