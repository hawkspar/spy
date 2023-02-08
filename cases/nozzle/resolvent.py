# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import numpy as np
from setup import *
from spyp import SPYP # Must be after setup
from spyb import SPYB
from mpi4py.MPI import COMM_WORLD as comm

ms=range(-5,5)
Sts=np.hstack((np.linspace(1,.2,20,endpoint=False),np.linspace(.2,.1,5,endpoint=False),np.linspace(.1,.01,10)))
Ss=[0,.4,1]
# Shorthands
Re=400000
for S in Ss:
	# Load baseflow
	spy = SPY(params,datapath,'baseflow',direction_map)
	spy.loadBaseflow(Re,S)
	spy.sanityCheckU()
	# Initialise resolvent toolbox (careful order sensitive)
	spyp=SPYP(params,datapath,"perturbations",direction_map)#,forcingIndicator)
	spyp.Re=Re
	d=dist(spyp)
	spyp.interpolateBaseflow(spy)
	spyp.sanityCheckU()
	for m in ms:
		boundaryConditionsPerturbations(spyp,m)
		# For efficiency, matrices assembled once per Sts
		spyp.assembleJNMatrices(m,d)
		spyp.assembleMRMatrices()
		# Resolvent analysis
		spyp.resolvent(1,Sts,Re,S,m)