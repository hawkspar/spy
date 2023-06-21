# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import ufl
import numpy as np
from sys import path
import dolfinx as dfx
from dolfinx.fem import Function

path.append('/home/shared/src')

from spy  import SPY, p0
from spyb import SPYB

base_mesh="baseflow"
pert_mesh="perturbations"

# Geometry parameters (nozzle)
R,H=1,20 # H as in OpenFOAM for BC compatibility

# /!\ OpenFOAM coherence /!\
Re=200000
h=1e-4
U_m,a=.05,6

# Easier standardisation across files
Ss_ref = np.linspace(0,1,6)
ms_ref = range(-5,6)
Sts_ref = np.linspace(0,.01,10)#np.hstack((np.linspace(0,.01,10,endpoint=False),np.linspace(.01,1,10)))

# Numerical Parameters
params = {"rp":.97,    #relaxation_parameter
		  "atol":1e-12, #absolute_tolerance
		  "rtol":1e-9, #DOLFIN_EPS does not work well
		  "max_iter":50}
data_path='nozzle' #folder for results
direction_map={'x':0,'r':1,'th':2}

spyb = SPYB(params,data_path,base_mesh,direction_map) # Must be first !

# Reference coherent with OpenFOAM
def nozzleTop(x): return 1+h+(x>.95)*(x-.95)/.05*h
def inletProfile(x): return np.tanh(a*(1-x[1]**2))
def coflowProfile(x):
	rt=(x[1]-1-h)/(H-1-h)
	return 2*U_m*(1-.5*rt)*rt

# Geometry
def symmetry(x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[1],0,			 params['atol']) # Axis of symmetry at r=0
def inlet(   x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[0],np.min(x[0]),params['atol'])*(x[1]<1+params['atol'])
def coflow(  x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[0],np.min(x[0]),params['atol'])*(x[1]>1+h-params['atol'])
def outlet(  x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[0],np.max(x[0]),params['atol']) # Right border
def top(     x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[1],np.max(x[1]),params['atol']) # Top boundary (assumed straight)
def nozzle(  x:ufl.SpatialCoordinate) -> np.ndarray: return (x[1]<nozzleTop(x[0])+params['atol'])*(R-params['atol']<x[1])*(x[0]<R+params['atol'])

# Necessary for resolvent stability at low St
def slope(d): return np.minimum(np.maximum(5*d+1,0),1)
x0,x1,x2=(1,1.1),(1.5,1),(12,3.5)
def line(x0,x1,x): return (x1[1]-x0[1])/(x1[0]-x0[0])*(x[0]-x0[0])+x0[1]-x[1]
def forcingIndicator(x): return (x[0]<x0[0])*slope(x0[1]-x[1])+\
				  (x[0]>=x0[0])*(x[0]<x1[0])*slope(line(x0,x1,x))+\
				  (x[0]>=x1[0])				*slope(line(x1,x2,x))

# Allows for more efficient change of S inside an iteration
class InletTangential:
	def __init__(self,S:float) -> None: self.S=S
	def __call__(self,x) -> np.array: return self.S*x[1]*inletProfile(x)

# u=(tanh 0 Srux) at inlet, u=(2rt(1-rt/2)Um 0 0) at coflow, u=0 at nozzle
# u_r, u_th=0 for symmetry axis (derived from momentum csv r th as r->0)
# However there are hidden BCs in the weak form !
# Because of the IPP, we have stress free BCs (pn=nu(gd U+gd U^T)n) everywhere by default
def boundaryConditionsBaseflow(spy:SPY,S:float) -> tuple:
	# Tool to pilot BC
	class_th=InletTangential(S)	
	# Dirichlet Boundary Conditions
	for (geo_indic,profile,dir) in [(coflow,coflowProfile,0),(inlet,inletProfile,0),(inlet,class_th,2)]: # CRITICAL that u_th is last
		subspace=spy.TH.sub(0).sub(dir)
		subspace_collapsed,_=subspace.collapse()
		u=Function(subspace_collapsed)
		u.interpolate(profile)
		# Degrees of freedom
		dofs = dfx.fem.locate_dofs_geometrical((subspace, subspace_collapsed), geo_indic)
		bcs  = dfx.fem.dirichletbc(u, dofs, subspace) # Same as OpenFOAM
		# Actual BCs
		spy.applyBCs(dofs[0],bcs)

	# Handle homogeneous boundary conditions
	spy.applyHomogeneousBCs([(inlet,['r']),(coflow,['r','th']),(nozzle,['x','r','th']),(symmetry,['r','th'])])
	return class_th, u

# u=0 at inlet, nozzle (linearise as baseflow)
# u_r, u_th=0 for symmetry axis if m=0, u_x=0 if |m|=1, u=0 else (derived from momentum csv r th as r->0)
# However there are hidden BCs in the weak form !
# Because of the IPP, we have stress free BCs everywhere by default
def boundaryConditionsPerturbations(spy:SPY,m:int) -> None:
	# Handle homogeneous boundary conditions
	homogeneous_boundaries=[(inlet,['x','r','th']),(coflow,['x','r','th']),(nozzle,['x','r','th'])]
	if 	     m ==0: homogeneous_boundaries.append((symmetry,['r','th']))
	elif abs(m)==1: homogeneous_boundaries.append((symmetry,['x'])) # Math checks out, but physically makes no sense to have r or theta components at r=0
	else:		    homogeneous_boundaries.append((symmetry,['x','r','th']))
	spy.applyHomogeneousBCs(homogeneous_boundaries)