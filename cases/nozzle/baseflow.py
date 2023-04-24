# coding: utf-8
"""
Created on Wed Sept  28 09:28:00 2022

@author: hawkspar
"""
from setup import *
from spyb import SPYB
from spy import loadStuff

spyb=SPYB(params,datapath,base_mesh,direction_map)

class_th,u_inlet_th=boundaryConditionsBaseflow(spyb,0)
# Highly viscous first step
spyb.loadBaseflow(1000,0)
spyb.Re=1000
spyb.baseflow(1000,0)
# No swirl
for Re in [1000,10000,100000,400000]:
	spyb.loadBaseflow(Re,0)
	spyb.Re=Re
	spyb.baseflow(Re,0)
"""spyb.loadBaseflow(400000,0,False)
spyb.Re=400000"""
# Swirl
for S in np.linspace(.1,1.3,13):
	class_th.S=S
	u_inlet_th.interpolate(class_th)
	loadStuff(spyb.nut_path,{"Re":Re,"S":S},spyb.Nu)
	spyb.baseflow(Re,S,save=False)
	spyb.smoothenU(1e-4,direction_map['r'])
	spyb.saveBaseflow(Re,S)
	U,_=spyb.Q.split()
	spyb.printStuff(spyb.print_path,f"u_Re={Re:d}_S={S:.1f}",U)