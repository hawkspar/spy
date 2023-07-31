# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import cProfile, pstats
from pstats import SortKey
from dolfinx.fem import FunctionSpace

from setup import *
from spyp import SPYP # Must be after setup

with cProfile.Profile() as pr:
	spyp=SPYP(params,data_path,pert_mesh,direction_map)

	FE_constant=ufl.FiniteElement("CG",spyp.mesh.ufl_cell(),1)
	indic_f = Function(FunctionSpace(spyp.mesh,FE_constant))
	indic_f.interpolate(forcingIndicator)
	spyp.assembleNMatrix()
	spyp.assembleMRMatrices(indic_f)

	for S in Ss_ref:
		# Load baseflow
		spyb.loadBaseflow(Re,S)
		# Initialise resolvent toolbox (careful order sensitive)
		spyp.Re=Re
		spyp.interpolateBaseflow(spyb)

		for m in ms_ref:
			boundaryConditionsPerturbations(spyp,m)
			# For efficiency, matrices assembled once per Sts
			spyp.assembleJMatrix(m)
			# Resolvent analysis
			spyp.resolvent(5,Sts_ref,Re,S,m)
	
	if p0:
		pr.dump_stats('stats')
		p = pstats.Stats('stats')
		p.sort_stats(SortKey.CUMULATIVE).print_stats(10)