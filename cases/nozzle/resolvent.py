# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import numpy as np
from setup import *
from spyp import SPYP # Must be after setup

ms=range(-5,6)
Sts=np.hstack((np.linspace(1,.2,20,endpoint=False),np.linspace(.2,.1,5,endpoint=False),np.linspace(.1,.01,10)))
# Load baseflow
spy = SPY(params,datapath,"baseflow",     direction_map)
spy.loadBaseflow(Re,S)
# Initialise resolvent toolbox (careful order sensitive)
spyp=SPYP(params,datapath,"perturbations",direction_map)
spyp.Re=Re
spyp.interpolateBaseflow(spy)
for m in ms:
	boundaryConditionsPerturbations(spyp,m)
	# For efficiency, matrices assembled once per Sts
	spyp.assembleJNMatrices(m)
	spyp.assembleMRMatrices()
	# Resolvent analysis
	spyp.resolvent(1,Sts,Re,S,m)