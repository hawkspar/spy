# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import cProfile, pstats
from pstats import SortKey

from setup import *
from spyp import SPYP # Must be after setup

Ss_ref=[1]
ms_ref=[2]
Sts_ref=np.linspace(0,1,26)

for mesh_app,app in [('',''),('_more','_345k'),('_morestill','_445k')]:
	with cProfile.Profile() as pr:
		spyp=SPYP(params,data_path,pert_mesh+mesh_app,direction_map,app)

		indic_f = Function(spyp.TH1)
		indic_f.interpolate(forcingIndicator)
		#spyp.printStuff(spyp.resolvent_path,"indic_f",indic_f)
		spyp.assembleMMatrix()
		spyp.assembleWBRMatrices(indic_f)

		for S in Ss_ref:
			# Load baseflow
			spyb.loadBaseflow(Re,S)
			# Initialise resolvent toolbox (careful order sensitive)
			spyp.Re=Re
			spyp.interpolateBaseflow(spyb)

			for m in ms_ref:
				boundaryConditionsPerturbations(spyp,m)
				# For efficiency, matrices assembled once per Sts
				spyp.assembleLMatrix(m)
				# Resolvent analysis
				spyp.resolvent(5,Sts_ref,Re,S,m)
		
		if p0:
			pr.dump_stats('stats')
			p = pstats.Stats('stats')
			p.sort_stats(SortKey.CUMULATIVE).print_stats(10)