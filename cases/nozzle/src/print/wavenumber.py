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
dat={'Re':Re,'m':-2,'St':7.3057e-03/2}
angles_dir=spyp.resolvent_path+"angles/"

# Approximate k as division of u
us=spyp.readMode("response",dat)
u,_,_=ufl.split(us)
FS = dfx.fem.FunctionSpace(spyp.mesh,ufl.FiniteElement("CG",spyp.mesh.ufl_cell(),2))
k2 = Function(FS)
k2.interpolate(dfx.fem.Expression(ufl.real(u.dx(0)/u/j(spyp.mesh)),FS.element.interpolation_points()))
spyp.printStuff(angles_dir,"k",k2)

"""# Compute phase
u,_,_=us.split()
u.x.array[:]=np.arctan2(u.x.array.imag,u.x.array.real)
u.x.scatter_forward()
spyp.printStuff(angles_dir,"phase",u)
# Approximate k as phase gradient
k1 = Function(FS)
k1.interpolate(dfx.fem.Expression(u.dx(0),FS.element.interpolation_points()))
spyp.printStuff(angles_dir,"k_phase",k1)"""

# Helix angles
#beta1=Function(FS)
beta2=Function(FS)
#beta1.interpolate(dfx.fem.Expression(-k1/dat['m'],FS.element.interpolation_points()))
beta2.interpolate(dfx.fem.Expression(-k2/dat['m'],FS.element.interpolation_points()))
#spyp.printStuff(angles_dir,"beta_phase",beta1)
spyp.printStuff(angles_dir,"beta",beta2)