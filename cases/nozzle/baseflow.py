# coding: utf-8
"""
Created on Wed Sept  28 09:28:00 2022

@author: hawkspar
"""
from setup import *
from spyb import SPYB
from spy import loadStuff

spyb=SPYB(params,datapath,"baseflow",direction_map)

class_th,u_inlet_th=boundaryConditionsBaseflow(spyb,0)
# Highly viscous first step
loadStuff(spyb.nut_path,{'S':0,'Re':1000},spyb.Nu)
#spyb.loadBaseflow(Re,S)
spyb.Re=1000
spyb.baseflow(1000,0,baseflowInit=baseflowInit)
# No swirl
for Re in [1000,10000,100000,400000]:
	loadStuff(spyb.nut_path,{'S':0,'Re':Re},spyb.Nu)
	#spyb.loadBaseflow(Re,S)
	spyb.Re=Re
	spyb.baseflow(Re,0)
"""spyb.loadBaseflow(400000,0)
spyb.Re=400000"""
# Swirl
for S in np.linspace(.1,1.6,16):
	class_th.S=S
	u_inlet_th.interpolate(class_th)
	loadStuff(spyb.nut_path,{'S':S,'Re':Re},spyb.Nu)
	spyb.baseflow(Re,S,save=True)
	spyb.smoothenU(1e-3)
	spyb.saveBaseflow(Re,S)
	U,_=spyb.Q.split()
	spyb.printStuff(spyb.print_path,f"u_Re={Re:d}_S={S:.1f}",U)