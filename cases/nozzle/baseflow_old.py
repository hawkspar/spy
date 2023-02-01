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
Re,nut=10,0
spyb.Re,spyb.Nu=Re,0
spyb.stabilise(0)
#spyb.loadBaseflow(Re,Re,0,True)
spyb.baseflow(Re,nut,0,d,baseflowInit=baseflowInit)
"""for Re in np.linspace(100,10000,30):
	spyb.Re=Re
	spyb.baseflow(int(Re),nut,0,d)"""
"""for Re in np.logspace(4,5,5,True,10):
	Re=int(Re)
	Ref(spyb,Re)
	loadStuff(spyb.nut_path,['S','nut','Re'],[0,Re,Re],spyb.Nu)
	spyb.baseflow(Re,Re,0,d)
#spyb.loadBaseflow(Re,nut,0,True)
for S in np.linspace(.1,1,6):
	class_th.S=S
	u_inlet_th.interpolate(class_th)
	loadStuff(spyb.nut_path,['S','nut','Re'],[S,nut,nut],spyb.Nu)
	print("# iterations :",spyb.baseflow(Re,nut,S,d))"""