# coding: utf-8
"""
Created on Wed Sept  28 09:28:00 2022

@author: hawkspar
"""
from setup import *
from spyb import SPYB
from copy import copy

Re0,S=100,0
dRe=1e-6
spyb=SPYB(params,datapath,"perturbations",direction_map)
boundaryConditionsBaseflow(spyb,S)
Ref(spyb,Re0)
spyb.stabilise(0)
spyb.baseflow(Re0,S,dist(spyb),baseflowInit=baseflowInit)

def march_Re(Re0,Re1):
	Q0=copy(spyb.Q)
	Ref(spyb,Re0+dRe)
	spyb.baseflow(Re0+dRe,S,dist(spyb),save=False)
	spyb.Q=Q0+(Q0-spyb.Q)/dRe*(Re1-Re1)
	n=spyb.baseflow(Re1,S,dist(spyb))
	_,P,Nu=spyb.Q.split()
	spyb.printStuff(spyb.print_path,f"p_S={S:d}_Re={Re1:d}", P)
	spyb.printStuff(spyb.print_path,f"nu_S={S:d}_Re={Re1:d}",Nu)
	return n

Re1=1000
while Re1<400000:
	n=march_Re(Re0,Re1)
	Re0=Re1
	if   n>=2: Re1*=2
	elif n>=5: Re1+=Re1
	else:	   Re1+=Re1/2