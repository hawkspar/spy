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

loadStuff(spyb.nut_path,{'S':0,'nut':1000,'Re':1000},spyb.Nu)
spyb.Re=1000
spyb.baseflow(1000,1000,0,d,baseflowInit=baseflowInit)
for Re in [10000,100000,400000]:
	loadStuff(spyb.nut_path,{'S':0,'nut':Re,'Re':Re},spyb.Nu)
	spyb.Re=Re
	spyb.baseflow(Re,Re,0,d)
for S in np.linspace(.2,1,5):
	class_th.S=S
	u_inlet_th.interpolate(class_th)
	loadStuff(spyb.nut_path,{'S':S,'nut':nut,'Re':nut},spyb.Nu)
	spyb.baseflow(nut,nut,S,d)