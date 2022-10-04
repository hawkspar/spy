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

from spy import SPY,loadStuff

# Geometry parameters (nozzle)
R=1

# /!\ OpenFOAM coherence /!\
Re=400000
S=0

# Numerical Parameters
params = {"rp":.95,    #relaxation_parameter
		  "atol":1e-9, #absolute_tolerance
		  "rtol":1e-6, #DOLFIN_EPS does not work well
		  "max_iter":100}
datapath='nozzle/' #folder for results
direction_map={'x':0,'r':1,'th':2}

# Geometry
def inlet( x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[0],0,		   params['atol']) # Left border
def outlet(x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[0],np.max(x[0]),params['atol']) # Right border
def top(   x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[1],np.max(x[1]),params['atol']) # Top (tilded) boundary
def nozzle(x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[1],R,		   params['atol'])*(x[0]<R)

def Ref(spy:SPY): return Re

def nutf(spy:SPY,S,Re):
	loadStuff(spy.nut_path+"complex/",['S','Re'],[S,Re],spy.nut.vector,spy.io)
	spy.nut.x.array[spy.nut.x.array<0]=0 # Enforce positive

# Baseflow (really only need DirichletBC objects) enforces :
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
	elif abs(m)==1: homogeneous_boundaries.append((spy.symmetry,['x']))
	else:		    homogeneous_boundaries.append((spy.symmetry,['x','r','th']))
	spy.applyHomogeneousBCs(homogeneous_boundaries)

def boundaryConditionsBaseflow(spy:SPY) -> None:
	# Compute DoFs
	sub_space_x=spy.TH.sub(0).sub(0)
	sub_space_x_collapsed,_=sub_space_x.collapse()

	u_inlet_x=Function(sub_space_x_collapsed)
	u_inlet_x.interpolate(lambda x: 10*np.tanh(6*(1-x[1]**2))*(x[1]<1)+
							  .5*np.tanh(6*(x[1]**2-1))*(x[1]>1))
	
	# Degrees of freedom
	dofs_inlet_x = dfx.fem.locate_dofs_geometrical((sub_space_x, sub_space_x_collapsed), inlet)
	bcs_inlet_x = dfx.fem.dirichletbc(u_inlet_x, dofs_inlet_x, sub_space_x) # Same as OpenFOAM

	# Actual BCs
	spy.applyBCs(dofs_inlet_x[0],[bcs_inlet_x]) # x=X entirely handled by implicit Neumann

	# Same for tangential
	sub_space_th=spy.TH.sub(0).sub(2)
	sub_space_th_collapsed,_=sub_space_th.collapse()

	# Modified vortex that goes to zero at top boundary
	u_inlet_th=Function(sub_space_th_collapsed)
	spy.inlet_azimuthal_velocity=InletAzimuthalVelocity(0)
	u_inlet_th.interpolate(spy.inlet_azimuthal_velocity)
	dofs_inlet_th = dfx.fem.locate_dofs_geometrical((sub_space_th, sub_space_th_collapsed), inlet)
	bcs_inlet_th = dfx.fem.dirichletbc(u_inlet_th, dofs_inlet_th, sub_space_th) # Same as OpenFOAM
	spy.applyBCs(dofs_inlet_th,[bcs_inlet_th])

	# Handle homogeneous boundary conditions
	spy.applyHomogeneousBCs([(nozzle,['x','r','th']),(spy.symmetry,['r','th'])])

class InletAzimuthalVelocity():
	def __init__(self, S): self.S = 0
	def __call__(self, x): return self.S*x[0]

def forcingIndicator(x): return np.isclose(x[1],R,.2)*(x[0]<R+.2)