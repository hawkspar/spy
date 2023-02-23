# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import numpy as np
from setup import *
from spyp import SPYP # Must be after setup
from dolfinx.fem import FunctionSpace

#ms=range(-2,3)
ms=[0]
Ss=[0]
#Sts=np.flip(np.hstack((np.linspace(1,.2,20,endpoint=False),np.linspace(.2,.1,5,endpoint=False),np.linspace(.1,.01,10))))
Sts=np.linspace(.1,.01,5)
for S in Ss:
	# Load baseflow
	spy = SPY(params,datapath,"baseflow",     direction_map)
	spy.loadBaseflow(Re,S)
	# Initialise resolvent toolbox (careful order sensitive)
	spyp=SPYP(params,datapath,"perturbations",direction_map)
	spyp.Re=Re
	spyp.interpolateBaseflow(spy)

	FE_constant=ufl.FiniteElement("DG",spyp.mesh.ufl_cell(),0)
	W = FunctionSpace(spyp.mesh,FE_constant)
	indic = Function(W)
	indic.interpolate(lambda x: forcing_indicator(x,))
	spyp.assembleMRMatrices(indic)

	for m in ms:
		boundaryConditionsPerturbations(spyp,m)
		# For efficiency, matrices assembled once per Sts
		spyp.assembleJNMatrices(m)
		# Resolvent analysis
		spyp.resolvent(1,Sts,Re,S,m)