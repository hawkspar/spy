# coding: utf-8
"""
Created on Wed Sept  28 09:28:00 2022

@author: hawkspar
"""
from setup import *
from spy import loadStuff

# Relevant parameters
Res=[1000,10000,100000,200000]
Ss =np.linspace(0,1,101)
compute_low_Re=True

if compute_low_Re:
	class_th,u_inlet_th=boundaryConditionsBaseflow(spyb,0)
	# No swirl
	spyb.loadBaseflow(Res[0],0)
	spyb.Re=Res[0]
	spyb.baseflow(Res[0],0)
	for Re in Res[1:]:
		loadStuff(spyb.nut_path,{"Re":Re,"S":0},spyb.Nu)
		spyb.Re=Re
		spyb.baseflow(Re,0)
else:
	spyb.loadBaseflow(Re,0,False)
	spyb.Re=Re
# Swirl
for S in Ss[1:]:
	class_th.S=S
	u_inlet_th.interpolate(class_th)
	loadStuff(spyb.nut_path,{"Re":Re,"S":S},spyb.Nu)
	spyb.baseflow(Re,S,save=False)
	spyb.smoothenU(1e-4,direction_map['r'])
	spyb.saveBaseflow(Re,S,print=True)