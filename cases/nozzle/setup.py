# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import ufl, sys
import numpy as np
import dolfinx as dfx

sys.path.append('/home/shared/src')

from spy import SPY
from spyb import SPYB

# Geometry parameters (nozzle)
R=1; L=60; H=15

# /!\ OpenFOAM coherence /!\
Re=400000
S=0

# Numerical Parameters
params = {"rp":.9,    #relaxation_parameter
		  "atol":1e-6, #absolute_tolerance
		  "rtol":1e-4, #DOLFIN_EPS does not work well
		  "max_iter":1000}
datapath='nozzle/' #folder for results
direction_map={'x':0,'r':1,'th':2}

class InletAzimuthalVelocity():
	def __init__(self, S): self.S = S
	def __call__(self, x):
		uth=0*x[1]
		msk=x[1]<R
		r=x[1][msk]/R
		uth[msk]=S*r*np.tanh(6*(1-r**2))
		return uth

def inflow(x):
	u=0*x[1]
	msk=x[1]<R
	u[~msk]=   np.tanh(6*((x[1][~msk]/R)**2-1))
	u[ msk]=10*np.tanh(6*(1-(x[1][ msk]/R)**2))
	return u

def baseflowInit(x):
	u=0*x
	u[0]=inflow(x)
	u[2]=InletAzimuthalVelocity(S)(x)
	return u

# Geometry
def inlet( x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[0],0,params['atol']) # Left border
def outlet(x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[0],L,params['atol']) # Right border
def top(   x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[1],H,params['atol']) # Top (tilded) boundary
def nozzle(x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[1],R,params['atol'])*(x[0]<R+params['atol'])

# Restriction on forcing area
#def forcingIndicator(x): return (x[0]<R+params['atol'])*(x[1]<R+params['atol'])

# Physical quantities
def Ref(spy:SPY): return Re
def nutf(spy:SPY,S:float): spy.loadStuff([S,Re],spy.nut_path,['S','Re'],spy.nut.vector)

# u_x=tanh, u_r=0 & u_th=S*r at inlet (velocity control)
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
	sub_space_x_collapsed=sub_space_x.collapse()

	u_i=dfx.Function(sub_space_x_collapsed)
	u_i.interpolate(lambda x: inflow(x))
	
	# Degrees of freedom
	dofs_inlet_x = dfx.fem.locate_dofs_geometrical((sub_space_x, sub_space_x_collapsed), inlet)
	bcs_inlet_x = dfx.DirichletBC(u_i, dofs_inlet_x, sub_space_x, np.float64) # u_x=tanh(6*(1-r)) at x=0

	# Actual BCs
	spyb.applyBCs(dofs_inlet_x,[bcs_inlet_x]) # outlet entirely handled by implicit Neumann
	
	# Handle homogeneous boundary conditions
	spyb.applyHomogeneousBCs([(inlet,['r','th']),(nozzle,['x','r','th']),(spyb.symmetry,['r','th'])])

# u=0 at inlet, nozzle & top (linearise as baseflow)
# u_r, u_th=0 for symmetry axis if m=0, u_x=0 if |m|=1, u=0 else (derived from momentum csv r th as r->0)
# However there are hidden BCs in the weak form !
# Because of the IPP, we have nu grad u.n=p n everywhere. This gives :
# d_ru_x=0 for symmetry axis (if not overwritten by above)
# d_xu_x/Re=p, d_xu_r=0, d_xu_th=0 at outflow (free flow)
def boundaryConditionsPerturbations(spy:SPY,m:int) -> None:
	# Handle homogeneous boundary conditions
	homogeneous_boundaries=[(inlet,['x','r','th']),(nozzle,['x','r','th']),(top,['x','r','th'])]
	if 	     m ==0: homogeneous_boundaries.append((spy.symmetry,['r','th']))
	#elif abs(m)==1: homogeneous_boundaries.append((spy.symmetry,['x']))
	else:		    homogeneous_boundaries.append((spy.symmetry,['x','r','th']))
	spy.applyHomogeneousBCs(homogeneous_boundaries)