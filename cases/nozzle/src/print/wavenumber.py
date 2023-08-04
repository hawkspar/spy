# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from sys import path

path.append('/home/shared/cases/nozzle/src/')

from setup import *
from spyp import SPYP
from helpers import j

# Shortcuts
spyp=SPYP(params,data_path,"perturbations",direction_map)
dats=[{'Re':Re,'S':S,'m':-2,'St':7.3057e-03/2} for S in np.linspace(0,1,6)]
source="response"
angles_dir=spyp.resolvent_path+source+"/angles/"

for dat in dats:
	save_str=f"_S={dat['S']}"
	# Approximate k as division of u
	us=spyp.readMode(source,dat)
	u,_,_=ufl.split(us)
	FS = dfx.fem.FunctionSpace(spyp.mesh,ufl.FiniteElement("CG",spyp.mesh.ufl_cell(),2))
	k = Function(FS)
	k.interpolate(dfx.fem.Expression(ufl.real(u.dx(0)/u/j(spyp.mesh)),FS.element.interpolation_points()))

	# Helix angles
	beta=Function(FS)
	beta.interpolate(dfx.fem.Expression(-k/dat['m'],FS.element.interpolation_points()))
	spyp.printStuff(angles_dir,"beta"+save_str,beta)

	# Maximum centrifugal strength
	spyb.loadBaseflow(Re,dat['S'],False)
	U,_=ufl.split(spyb.Q)
	FS = dfx.fem.FunctionSpace(spyb.mesh,ufl.FiniteElement("CG",spyb.mesh.ufl_cell(),2))
	g = Function(FS)
	g.interpolate(dfx.fem.Expression(U[2]/spyb.r/U[0],FS.element.interpolation_points()))
	spyb.printStuff(angles_dir,"centrifuge_growth"+save_str,g)
	g.interpolate(dfx.fem.Expression((U[2]/spyb.r).dx(1)/U[0].dx(1),FS.element.interpolation_points()))
	spyb.printStuff(angles_dir,"centrifuge_growth_2"+save_str,g)