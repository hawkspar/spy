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

from spy  import SPY
from spyb import SPYB

# Geometry parameters (validation legacy)
x_max=40; r_max=10
x_p=-5;   R=1

S,Re=0,1000

# Numerical parameters
params = {"rp":.99,    #relaxation_parameter
		  "atol":1e-6, #absolute_tolerance
		  "rtol":1e-9, #DOLFIN_EPS does not work well
		  "max_iter":1000}
datapath='garnaud' #folder for results
direction_map={'x':0,'r':1,'th':2}

def baseflowInit(x):
	u=0*x
	u[0,x[1]<1]=np.tanh(5*(1-x[1][x[1]<1]))
	return u

# Geometry
def symmetry(x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[1],0,	  params['atol']) # Axis of symmetry at r=0
def inlet( x:ufl.SpatialCoordinate)   -> np.ndarray: return np.isclose(x[0],x_p,  params['atol']) # Left border
def outlet(x:ufl.SpatialCoordinate)   -> np.ndarray: return np.isclose(x[0],x_max,params['atol']) # Right border
def top(   x:ufl.SpatialCoordinate)   -> np.ndarray: return np.isclose(x[1],r_max,params['atol']) # Top boundary at r=R
def wall(  x:ufl.SpatialCoordinate)   -> np.ndarray: return (x[0]<params['atol'])*(x[1]>R-params['atol']) # Walls

# Restriction on forcing area
def forcingIndicator(x): return x[0]<-params['atol']

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
def boundaryConditionsBaseflow(spyb:SPYB) -> None:
	# Compute DoFs
	sub_space_x=spyb.TH.sub(0).sub(0)
	sub_space_x_collapsed,_=sub_space_x.collapse()

	u_inlet_x=Function(sub_space_x_collapsed)
	u_inlet_x.interpolate(lambda x: np.tanh(5*(1-x[1])))
	
	# Degrees of freedom
	dofs_inlet_x = dfx.fem.locate_dofs_geometrical((sub_space_x, sub_space_x_collapsed), inlet)
	bcs_inlet_x = dfx.fem.dirichletbc(u_inlet_x, dofs_inlet_x, sub_space_x) # u_x=tanh(5*(1-r)) at x=0

	# Actual BCs
	spyb.applyBCs(dofs_inlet_x[0],bcs_inlet_x) # x=X entirely handled by implicit Neumann
	
	# Handle homogeneous boundary conditions
	spyb.applyHomogeneousBCs([(inlet,['r','th']),(wall,['x','r','th']),(symmetry,['r','th'])])

# Baseflow (really only need DirichletBC objects) enforces :
# u=0 at inlet (linearise as baseflow)
# u_r, u_th=0 for symmetry axis if m=0, u_x=0 if |m|=1, u=0 else (derived from momentum csv r th as r->0)
# u_r=0, u_th=0 at top (Meliga paper, no slip)
# However there are hidden BCs in the weak form !
# Because of the IPP, we have nu grad u.n=p n everywhere. This gives :
# d_ru_x=0 for symmetry axis (if not overwritten by above)
# d_xu_x/Re=p, d_xu_r=0, d_xu_th=0 at outflow (free flow)
# d_ru_x=0 at top (Meliga paper, no slip)
def boundaryConditionsPerturbations(spy:SPY, m:int) -> None:
	# Handle homogeneous boundary conditions
	homogeneous_boundaries=[(inlet,['x','r','th']),(wall,['x','r','th'])]
	if 	     m ==0: homogeneous_boundaries.append((symmetry,['r','th']))
	elif abs(m)==1: homogeneous_boundaries.append((symmetry,['x']))
	else:		    homogeneous_boundaries.append((symmetry,['x','r','th']))
	spy.applyHomogeneousBCs(homogeneous_boundaries)