# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import cProfile, pstats
from pstats import SortKey
from dolfinx.fem import FunctionSpace
from mpi4py.MPI import COMM_WORLD as comm

from setup import *
from spyp import SPYP # Must be after setup

from helpers import loadStuff

with cProfile.Profile() as pr:
	spyp=SPYP(params,data_path,pert_mesh,direction_map)

	FE_constant=ufl.FiniteElement("CG",spyp.mesh.ufl_cell(),1)
	W = FunctionSpace(spyp.mesh,FE_constant)
	indic_q,indic_f = Function(W),Function(W)
	#indic_q.interpolate(lambda x: slope(x[0]-1))
	indic_f.interpolate(forcingIndicator)
	#spyp.printStuff('./','indic_q',indic_q)
	#spyp.printStuff('./','indic_f',indic_f)
	spyp.assembleNMatrix()#indic_q)
	spyp.assembleMRMatrices(indic_f)

	for S in Ss_ref:
		# Load baseflow
		#spyb.loadBaseflow(Re,S)
		loadStuff(spyb.q_path,  {'Re':Re,'S':S},spyb.Q)
		loadStuff(spyb.nut_path,{'Re':Re,'S':0},spyb.Nu)
		# Initialise resolvent toolbox (careful order sensitive)
		spyp.Re=Re
		spyp.interpolateBaseflow(spyb)

		for m in ms_ref:
			boundaryConditionsPerturbations(spyp,m)
			# For efficiency, matrices assembled once per Sts
			spyp.assembleJMatrix(m)
			# Resolvent analysis
			spyp.resolvent(3,Sts_ref,Re,S,m)
	
	if comm.rank==0:
		pr.dump_stats('stats')
		p = pstats.Stats('stats')
		p.sort_stats(SortKey.CUMULATIVE).print_stats(10)