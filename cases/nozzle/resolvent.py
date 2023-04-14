# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from setup import *
from spyp import SPYP # Must be after setup
import cProfile, pstats
from pstats import SortKey
from dolfinx.fem import FunctionSpace
from mpi4py.MPI import COMM_WORLD as comm

with cProfile.Profile() as pr:
	Ss=np.linspace(0,1.2,7)
	ms=range(-3,4)
	Sts=np.linspace(.05,2,10)#np.hstack((np.linspace(.05,.1,3,endpoint=False),np.linspace(.1,2,10)))
	spy = SPY(params,datapath,"baseflow",     direction_map) # Must be first !
	spyp=SPYP(params,datapath,"perturbations",direction_map)

	FE_constant=ufl.FiniteElement("CG",spyp.mesh.ufl_cell(),1)
	W = FunctionSpace(spyp.mesh,FE_constant)
	indic = Function(W)
	indic.interpolate(forcing_indicator)
	spyp.printStuff('./','indic',indic)
	spyp.assembleMRMatrices(indic)

	for S in Ss:
		# Load baseflow
		spy.loadBaseflow(Re,S)
		# Initialise resolvent toolbox (careful order sensitive)
		spyp.Re=Re
		spyp.interpolateBaseflow(spy)

		for m in ms:
			boundaryConditionsPerturbations(spyp,m)
			# For efficiency, matrices assembled once per Sts
			spyp.assembleJNMatrices(m)
			# Resolvent analysis
			spyp.resolvent(3,Sts,Re,S,m)
	if comm.rank==0:
		pr.dump_stats('stats')
		p = pstats.Stats('stats')
		p.sort_stats(SortKey.CUMULATIVE).print_stats(10)