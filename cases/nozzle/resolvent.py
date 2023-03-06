# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import numpy as np
from setup import *
from spyp import SPYP # Must be after setup
from dolfinx.fem import FunctionSpace

#ms=range(-5,6)
ms=range(6)
Ss=[0,1,.4]
Sts=np.linspace(.05,2,30)

for S in Ss:
	# Load baseflow
	spy = SPY(params,datapath,"baseflow",     direction_map)
	spy.loadBaseflow(Re,S)
	# Initialise resolvent toolbox (careful order sensitive)
	spyp=SPYP(params,datapath,"perturbations",direction_map)
	spyp.Re=Re
	spyp.interpolateBaseflow(spy)

	FE_constant=ufl.FiniteElement("CG",spyp.mesh.ufl_cell(),1)
	W = FunctionSpace(spyp.mesh,FE_constant)
	indic = Function(W)
	indic.interpolate(forcing_indicator)
	spyp.printStuff('./','indic',indic)
	spyp.assembleMRMatrices(indic)

	for m in ms:
		boundaryConditionsPerturbations(spyp,m)
		# For efficiency, matrices assembled once per Sts
		spyp.assembleJNMatrices(m)
		# Resolvent analysis
		spyp.resolvent(1,Sts,Re,S,m)
	# Destroy them before next loop
	del spy,spyp