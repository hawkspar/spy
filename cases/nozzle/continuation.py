# coding: utf-8
"""
Created on Wed Sept  28 09:28:00 2022

@author: hawkspar
"""
from setup import *
from spyb import SPYB
from copy import copy

Re,S=100,0
spyb=SPYB(params,datapath,"baseflow",direction_map)
boundaryConditionsBaseflow(spyb,S)
Ref(spyb,Re)
spyb.stabilise(0)
# Shorthands
d=dist(spyb)
weak_bcs  =weakBoundaryConditions(spyb,spyb.trial,spyb.Q,spyb.test)
weak_bcs_e=weakBoundaryConditions(spyb,spyb.qtrial,spyb.Qe,spyb.qtest)
spyb.baseflow(Re,S,d,weak_bcs,baseflowInit=baseflowInit)

h=1000
dRe=1e-9
while Re<400000:
	Q0=copy(spyb.Q)
	# Predictor
	Ref(spyb,Re+dRe)
	spyb.baseflow(Re+dRe,S,d,weak_bcs,save=False)
	dQ=Q0-spyb.Q
	spyb.Q=Q0+dQ/dRe*h
	Re,n=spyb.corrector(Q0,dQ,dRe,h,Re,S,d,weak_bcs_e)
	U,P,Nu=spyb.Q.split()
	spyb.printStuff(spyb.print_path,f"u_S={S:d}_Re={Re:d}", U)
	spyb.printStuff(spyb.print_path,f"p_S={S:d}_Re={Re:d}", P)
	spyb.printStuff(spyb.print_path,f"nu_S={S:d}_Re={Re:d}",Nu)
	# Cheap step size adaptation
	if   n<=2: h*=1.5
	elif n<=3: h*=1.2
	else:	   h/=2