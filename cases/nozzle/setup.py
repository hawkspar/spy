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

from spy import SPY

# Geometry parameters (nozzle)
R,H,L=1,15,50.5

# /!\ OpenFOAM coherence /!\
S,Re=1,400000
h=1e-4
U_m,a=.05,6

# Numerical Parameters
params = {"rp":.95,    #relaxation_parameter
		  "atol":1e-6, #absolute_tolerance
		  "rtol":1e-4, #DOLFIN_EPS does not work well
		  "max_iter":50}
datapath='nozzle/' #folder for results
direction_map={'x':0,'r':1,'th':2}

# Reference coherent with OpenFOAM
def nozzle_top(x): return R+h+(x>.95*R)*(x-.95*R)/.05/R*h
def inletProfile(x): return np.tanh(a*(1-x[1]**2))			  *(x[1]<1)+\
		 				    2*U_m*(1-.5*(x[1]-1)/H)*(x[1]-1)/H*(x[1]>1)

# Geometry
def symmetry(x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[1],0,			 params['atol']) # Axis of symmetry at r=0
def inlet(   x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[0],0,		     params['atol']) # Left border
def outlet(  x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[0],np.max(x[0]),params['atol']) # Right border
def top(     x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[1],np.max(x[1]),params['atol']) # Top boundary (assumed straight)
def nozzle(  x:ufl.SpatialCoordinate) -> np.ndarray: return (x[1]<nozzle_top(x[0])+params['atol'])*(R-params['atol']<x[1])*(x[0]<R+params['atol'])

# Necessary for resolvent stability at low St
def slope(x,xp,s=0): return np.minimum(np.maximum(5*(-1)**s*(xp-x)+1,0),1)
def forcing_indicator(x): return ((x[1]<=1+params['atol'])+(x[1]>1+params['atol'])*slope(x[1],1+x[0]*.5/5))*slope(x[0],2)

# Simplistic profile to initialise Newton
def baseflowInit(x):
	u=0*x
	u[0]=inletProfile(x)
	return u

# Allows for more efficient change of S inside an iteration
class inletTangential:
	def __init__(self,S:float) -> None: self.S=S
	def __call__(self,x) -> np.array: return self.S*x[1]*inletProfile(x)*(x[1]<1)

def boundaryConditionsBaseflow(spy:SPY,S) -> None:
	# Compute DoFs
	sub_space_x=spy.TH.sub(0).sub(0)
	sub_space_x_collapsed,_=sub_space_x.collapse()

	u_inlet_x=Function(sub_space_x_collapsed)
	u_inlet_x.interpolate(inletProfile)
	# Degrees of freedom
	dofs_inlet_x = dfx.fem.locate_dofs_geometrical((sub_space_x, sub_space_x_collapsed), inlet)
	bcs_inlet_x = dfx.fem.dirichletbc(u_inlet_x, dofs_inlet_x, sub_space_x) # Same as OpenFOAM

	# Actual BCs
	spy.applyBCs(dofs_inlet_x[0],bcs_inlet_x) # x=X entirely handled by implicit Neumann

	# Same for tangential
	sub_space_th=spy.TH.sub(0).sub(2)
	sub_space_th_collapsed,_=sub_space_th.collapse()
	u_inlet_th=Function(sub_space_th_collapsed)
	class_th=inletTangential(S)
	u_inlet_th.interpolate(class_th)
	dofs_inlet_th = dfx.fem.locate_dofs_geometrical((sub_space_th, sub_space_th_collapsed), inlet)
	bcs_inlet_th = dfx.fem.dirichletbc(u_inlet_th, dofs_inlet_th, sub_space_th) # Same as OpenFOAM
	spy.applyBCs(dofs_inlet_th[0],bcs_inlet_th)

	# Handle homogeneous boundary conditions
	spy.applyHomogeneousBCs([(inlet,['r']),(nozzle,['x','r','th']),(symmetry,['r','th'])])
	return class_th,u_inlet_th

# Baseflow (really only need DirichletBC objects) enforces :
# u=0 at inlet, nozzle & top (linearise as baseflow)
# u_r, u_th=0 for symmetry axis if m=0, u_x=0 if |m|=1, u=0 else (derived from momentum csv r th as r->0)
# However there are hidden BCs in the weak form !
# Because of the IPP, we have stress free BCs everywhere by default
def boundaryConditionsPerturbations(spy:SPY,m:int) -> None:
	# Handle homogeneous boundary conditions
	homogeneous_boundaries=[(inlet,['x','r','th']),(nozzle,['x','r','th'])]
	if 	     m ==0: homogeneous_boundaries.append((symmetry,['r','th']))
	elif abs(m)==1: homogeneous_boundaries.append((symmetry,['x']))
	else:		    homogeneous_boundaries.append((symmetry,['x','r','th']))
	spy.applyHomogeneousBCs(homogeneous_boundaries)