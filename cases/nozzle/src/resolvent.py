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
	spyp=SPYP(params,data_path,pert_mesh,direction_map)

	FE_constant=ufl.FiniteElement("CG",spyp.mesh.ufl_cell(),1)
	W = FunctionSpace(spyp.mesh,FE_constant)
	indic = Function(W)
	indic.interpolate(forcingIndicator)
	#spyp.printStuff('./','indic',indic)
	spyp.assembleMRMatrices(indic)

	for S in Ss_ref:
		# Load baseflow
		spyb.loadBaseflow(Re,S)
		# Initialise resolvent toolbox (careful order sensitive)
		spyp.Re=Re
		spyp.interpolateBaseflow(spyb)

		for m in ms_ref:
			boundaryConditionsPerturbations(spyp,m)
			# For efficiency, matrices assembled once per Sts
			spyp.assembleJNMatrices(m)
			# Resolvent analysis
			spyp.resolvent(3,Sts_ref,Re,S,m)
	if comm.rank==0:
		pr.dump_stats('stats')
		p = pstats.Stats('stats')
		p.sort_stats(SortKey.CUMULATIVE).print_stats(10)