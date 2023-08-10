# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from sys import path

path.append('/home/shared/cases/nozzle/src/')

from setup import *
from spyp import SPYP

# Shortcuts
spyp=SPYP(params,data_path,"perturbations",direction_map)
angles_dir=spyp.resolvent_path+"angles/"

# Load baseflow, extract 2 main components
spyb.loadBaseflow(Re,1,False)
U,_=spyb.Q.split()
Ux,_,Ut=ufl.split(U)
# Gradient of these components along r
FS_v = dfx.fem.FunctionSpace(spyb.mesh,ufl.VectorElement("CG",spyb.mesh.ufl_cell(),2))
dU = Function(FS_v)
dU.interpolate(dfx.fem.Expression(ufl.as_vector([Ux.dx(1),Ut.dx(1)]),FS_v.element.interpolation_points()))

# Actual velocity angle in plane
FS = dfx.fem.FunctionSpace(spyb.mesh,ufl.FiniteElement("CG",spyb.mesh.ufl_cell(),2)) # != mesh !!!
ang = Function(FS)
dUx,dUt=ufl.split(dU)
ang.interpolate(dfx.fem.Expression(ufl.atan_2(dUt,dUx),FS.element.interpolation_points()))
spyb.printStuff(angles_dir,"dU_angle",ang)