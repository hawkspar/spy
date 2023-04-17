# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import ufl, sys
import numpy as np

sys.path.append('/home/shared/src')

from spy import SPY

base_mesh="baseflow"
pert_mesh="perturbations"

# Geometry parameters (nozzle)
R,H,L=1,15,50.5

# /!\ OpenFOAM coherence /!\
S,Re=1,400000
h=1e-4
U_m,a=.05,6

# Numerical Parameters
params = {"rp":.95,    #relaxation_parameter
		  "atol":1e-9, #absolute_tolerance
		  "rtol":1e-6, #DOLFIN_EPS does not work well
		  "max_iter":100}
datapath='no_nozzle' #folder for results
direction_map={'x':0,'r':1,'th':2}

spy_nozzle = SPY(params,"nozzle","baseflow",direction_map) # Important ! Mesh loading order is critical

# Reference coherent with OpenFOAM
def nozzle_top(x): return R+h+(x>.95*R)*(x-.95*R)/.05/R*h
def inletProfile(x): return np.tanh(a*(1-x[1]**2))			  *(x[1]<1)+\
		 				    2*U_m*(1-.5*(x[1]-1)/H)*(x[1]-1)/H*(x[1]>1)

# Geometry
def symmetry(x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[1],0,			 params['atol']) # Axis of symmetry at r=0
def inlet(   x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[0],1,		     params['atol']) # Left border
def outlet(  x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[0],np.max(x[0]),params['atol']) # Right border
def top(     x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[1],np.max(x[1]),params['atol']) # Top boundary (assumed straight)

# Necessary for resolvent stability at low St
def slope(x,xp,s=0): return np.minimum(np.maximum(5*(-1)**s*(xp-x)+1,0),1)
def forcing_indicator(x): return slope(x[1],1+x[0]/10)*slope(x[1],2)

# Simplistic profile to initialise Newton
def baseflowInit(x):
	u=0*x
	u[0]=inletProfile(x)
	return u

# Baseflow (really only need DirichletBC objects) enforces :
# u=0 at inlet, nozzle & top (linearise as baseflow)
# u_r, u_th=0 for symmetry axis if m=0, u_x=0 if |m|=1, u=0 else (derived from momentum csv r th as r->0)
# However there are hidden BCs in the weak form !
# Because of the IPP, we have stress free BCs everywhere by default
def boundaryConditionsPerturbations(spy:SPY,m:int) -> None:
	# Handle homogeneous boundary conditions
	homogeneous_boundaries=[(inlet,['x','r','th'])]
	if 	     m ==0: homogeneous_boundaries.append((symmetry,['r','th']))
	elif abs(m)==1: homogeneous_boundaries.append((symmetry,['x']))
	else:		    homogeneous_boundaries.append((symmetry,['x','r','th']))
	spy.applyHomogeneousBCs(homogeneous_boundaries)