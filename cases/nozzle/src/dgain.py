# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from dolfinx.fem import FunctionSpace

from setup import *
from spyp import SPYP # Must be after setup
from helpers import findStuff

dats=[{"S":1,"m":-2,"St":0}]

h=1e-4
Sts_0_ref, Sts_ref = [0], []
for St_0 in Sts_0_ref:
	Sts_ref.append(St_0-h)
	Sts_ref.append(St_0+h)

spyp=SPYP(params,data_path,pert_mesh,direction_map)

FE_constant=ufl.FiniteElement("CG",spyp.mesh.ufl_cell(),1)
indic_f = Function(FunctionSpace(spyp.mesh,FE_constant))
indic_f.interpolate(forcingIndicator)
spyp.assembleNMatrix()
spyp.assembleMRMatrices(indic_f)

for dat in dats:
	St=dat['St']
	# Load baseflow
	spyb.loadBaseflow(Re,dat['S'])
	# Initialise resolvent toolbox (careful order sensitive)
	spyp.Re=Re
	spyp.interpolateBaseflow(spyb)

	boundaryConditionsPerturbations(spyp,dat['m'])
	# For efficiency, matrices assembled once per Sts
	spyp.assembleJMatrix(dat['m'])
	# Resolvent analysis
	spyp.resolvent(1,[St-h,St+h],Re,dat['S'],dat['m'])

	if p0:
		dat['St']=St+h
		gph_file=findStuff(spyp.resolvent_path+"gains/txt/",dat,distributed=False)
		gph=np.loadtxt(gph_file)
		dat['St']=St-h
		gmh_file=findStuff(spyp.resolvent_path+"gains/txt/",dat,distributed=False)
		gmh=np.loadtxt(gmh_file)

		print("Derivative around ",St,":",(gph-gmh)/2/h)