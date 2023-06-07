# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import ufl, sys
import numpy as np
import dolfinx as dfx
from dolfinx.fem import Function

sys.path.append('/home/shared/src')

from spy import SPY, p0
from spyb import SPYB

# Geometry parameters
x_max,r_max=70,10
l=50
x_lim,r_lim=x_max+l,r_max+l

# Physical Parameters
Re=200
Re_s=.1

# Numerical parameters
params = {"rp":.97,    #relaxation_parameter
		  "atol":1e-12, #absolute_tolerance
		  "rtol":1e-9, #DOLFIN_EPS does not work well
		  "max_iter":200}
datapath='meliga' #folder for results
direction_map={'x':0,'r':1,'th':2}

# Geometry
def sym(   x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[1],0,		   params['atol']) # Axis of symmetry at r=0
def inlet( x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[0],0,		   params['atol']) # Left border
def outlet(x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[0],np.max(x[0]),params['atol']) # Right border
def top(   x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[1],np.max(x[1]),params['atol']) # Top boundary at r=R

# Damped Reynolds number
def csi(a,b): return .5*(1+np.tanh(4*np.tan(np.pi*(np.abs(a-b)/l-.5))))
def sponged_Reynolds(x):
	Rem=np.ones_like(x[0])*Re
	x_ext=x[0]>x_max
	# min necessary to prevent spurious jumps because of mesh approximations
	Rem[x_ext]=Re		 +(Re_s-Re) 	   *csi(np.minimum(x[0][x_ext],x_lim),x_max)
	r_ext=x[1]>r_max
	Rem[r_ext]=Rem[r_ext]+(Re_s-Rem[r_ext])*csi(np.minimum(x[1][r_ext],r_lim),r_max)
	return Rem

# Handler
spyb=SPYB(params,datapath,"validation",direction_map)
spyb.Re=np.inf
spyb.Nu.interpolate(lambda x: 1/sponged_Reynolds(x))

# Grabovski-Berger vortex with final slope
def grabovski_berger(r) -> np.ndarray:
	psi=(r_lim-r)/l/r_max # Linear descent to zero at the top
	msk=r<=1
	psi[msk]=r[msk]*(2-r[msk]**2)
	msk=(r>1)*(r<r_max)
	psi[msk]=1/r[msk]
	return psi

class InletAzimuthalVelocity:
	def __init__(self, S): self.S = S
	def __call__(self, x): return self.S*grabovski_berger(x[1])

# Baseflow (really only need DirichletBC objects) enforces :
# u_x=1, u_r=0 & u_th=gb at inlet (velocity control)
# u_r=0, u_th=0 for symmetry axis (derived from mass csv as r->0)
# Nothing at outlet
# u_r=0, u_th=0 at top (Meliga paper, no slip)
# However there are hidden BCs in the weak form !
# Because of the IPP, we have nu grad u.n=p n everywhere. This gives :
# d_ru_x=0 for symmetry axis (momentum csv r as r->0)
# d_xu_x/Re=p, d_xu_r=0, d_xu_th=0 at outlet (free flow)
# d_ru_x=0 at top (Meliga paper, no slip)
def boundaryConditionsBaseflow(spyb:SPYB,S:float) -> tuple:	
	# Actual BCs
	dofs_inlet_x, bcs_inlet_x = spyb.constantBC('x',inlet,1) # u_x =1
	spyb.applyBCs(dofs_inlet_x,bcs_inlet_x)	

	# Compute DoFs
	sub_space_th=spyb.TH.sub(0).sub(2)
	sub_space_th_collapsed,_=sub_space_th.collapse()

	u_inlet_th=Function(sub_space_th_collapsed)
	class_th=InletAzimuthalVelocity(S)
	u_inlet_th.interpolate(class_th)
	
	# Degrees of freedom
	dofs_inlet_th = dfx.fem.locate_dofs_geometrical((sub_space_th, sub_space_th_collapsed), inlet)
	bcs_inlet_th = dfx.fem.dirichletbc(u_inlet_th, dofs_inlet_th, sub_space_th) # u_th=S*psi(r) at x=0
	spyb.applyBCs(dofs_inlet_th[0],bcs_inlet_th)

	# Handle homogeneous boundary conditions
	spyb.applyHomogeneousBCs([(inlet,['r']),(top,['r','th']),(sym,['r','th'])])
	boundaries = [(1, lambda x: top(x)*sym(x)),(2, outlet)]
	return class_th,u_inlet_th,boundaries

# Baseflow (really only need DirichletBC objects) enforces :
# u=0 at inlet (linearise as baseflow)
# u_r, u_th=0 for symmetry axis if m=0, u_x=0 if |m|=1, u=0 else (derived from momentum csv r th as r->0)
# u_r=0, u_th=0 at top (Meliga paper, no slip)
# However there are hidden BCs in the weak form !
# Because of the IPP, we have nu grad u.n=p n everywhere. This gives :
# d_ru_x=0 for symmetry axis (if not overwritten by above)
# d_xu_x/Re=p, d_xu_r=0, d_xu_th=0 at outflow (free flow)
# d_ru_x=0 at top (Meliga paper, no slip)
def boundaryConditionsPerturbations(spy:SPY,m:int) -> None:
	# Handle homogeneous boundary conditions
	homogeneous_boundaries=[(inlet,['x','r','th']),(top,['r','th'])]
	if 	     m ==0: homogeneous_boundaries.append((sym,['r','th']))
	elif abs(m)==1: homogeneous_boundaries.append((sym,['x']))
	else:		    homogeneous_boundaries.append((sym,['x','r','th']))
	spy.applyHomogeneousBCs(homogeneous_boundaries)