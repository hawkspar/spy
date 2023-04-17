# coding: utf-8
"""
Created on Wed Sept  28 09:28:00 2022

@author: hawkspar
"""
from setup import *
from spyb import SPYB
from spy import loadStuff

spyb=SPYB(params,datapath,base_mesh,direction_map)

u_inlet_th,u_inlet_x=boundaryConditionsBaseflow(spyb,0)
# Highly viscous first step
loadStuff(spyb.nut_path,{'S':0,'Re':1000},spyb.Nu)
spyb.Re=1000
spyb.baseflow(1000,0,baseflowInit=spy_nozzle.Q.split()[0])
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
	spy_nozzle.loadBaseflow(Re,S,False)
	U_nozzle,_ = spy_nozzle.Q.split()
	u_inlet_x.interpolate( lambda x: spy_nozzle.eval(U_nozzle[0],x))
	u_inlet_th.interpolate(lambda x: spy_nozzle.eval(U_nozzle[2],x))
	loadStuff(spyb.nut_path,{'S':S,'Re':Re},spyb.Nu)
	spyb.baseflow(Re,S,save=True)
	spyb.smoothenU(1e-3)
	spyb.saveBaseflow(Re,S)
	U,_=spyb.Q.split()
	spyb.printStuff(spyb.print_path,f"u_Re={Re:d}_S={S:.1f}",U)