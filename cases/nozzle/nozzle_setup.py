# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import ufl, sys
import numpy as np

sys.path.append('/home/shared/src')

from spy import SPY

# Geometry parameters (nozzle)
R=1;  L=100
h=15; H=20

# Numerical Parameters
params = {"rp":.99,    #relaxation_parameter
		  "atol":1e-6, #absolute_tolerance
		  "rtol":1e-9, #DOLFIN_EPS does not work well
		  "max_iter":100}

datapath='nozzle/' #folder for results

# Geometry
def inlet( x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[0],0,			 params['atol']) # Left border
def outlet(x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[0],L,			 params['atol']) # Right border
def top(   x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[1],h+(H-h)/L*x[0],params['atol']) # Top (tilded) boundary
def nozzle(x:ufl.SpatialCoordinate) -> np.ndarray: return np.logical_and(np.isclose(x[1],R,params['atol']),x[0]<R)

def Ref(x): return 1e4

def nutf(spy:SPY,S:float): spy.loadStuff(S,"last_nut.dat",spy.nut_path,6,spy.nut.vector)

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
	spy.dofps = np.empty(0,dtype=np.int32)
	for tup in homogeneous_boundaries:
		marker,directions=tup
		for direction in directions:
			dofs, _=spy.constantBC(direction,marker)
			spy.dofps=np.union1d(spy.dofps,dofs)