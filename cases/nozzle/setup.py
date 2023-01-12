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
R=1

# /!\ OpenFOAM coherence /!\
S,nut,Re=1,400000,400000
h=2.5e-4

# Numerical Parameters
params = {"rp":.9,    #relaxation_parameter
		  "atol":1e-9, #absolute_tolerance
		  "rtol":1e-6, #DOLFIN_EPS does not work well
		  "max_iter":500}
datapath='nozzle/' #folder for results
direction_map={'x':0,'r':1,'th':2}

def nozzle_top(x): return R+(1-x/R)*h

# Geometry
def inlet(   x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[0],0,		   params['atol']) # Left border
def outlet(  x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[0],np.max(x[0]),params['atol']) # Right border
def top(     x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[1],np.max(x[1]),params['atol']) # Top (tilded) boundary
def nozzle(  x:ufl.SpatialCoordinate) -> np.ndarray: return (x[1]<nozzle_top(x[0])+params['atol'])*(R-params['atol']<x[1])*(x[0]<R+params['atol'])
def symmetry(x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[1],0,params['atol']) # Axis of symmetry at r=0

def dist(spy:SPY):
	x,r=ufl.SpatialCoordinate(spy.mesh)[0],spy.r
	return ufl.sqrt((r-1)**2 + ufl.conditional(ufl.ge(x,R),(x-R)**2,0))

def Ref(spy:SPY,Re): spy.Re=Re

def forcingIndicator(x): return (x[1]<nozzle_top(x[0]))*(x[0]<1)+(x[1]<1+x[0]/3)*(x[0]>=1)*(x[0]<3)+(x[1]<2)*(x[0]>=3)

def baseflowInit(x):
	u=0*x
	u[0,x[1]<1]=np.tanh(6*(1-x[1][x[1]<1]))
	return u

def boundaryConditionsBaseflow(spy:SPY,S) -> None:
	# Compute DoFs
	sub_space_x=spy.FS.sub(0).sub(0)
	sub_space_x_collapsed,_=sub_space_x.collapse()

	u_inlet_x=Function(sub_space_x_collapsed)
	u_inlet_x.interpolate(lambda x: np.tanh(6*(1-x[1]**2))*(x[1]<1)+
							    .01*np.tanh(6*(x[1]**2-1))*(x[1]>1))
	
	# Degrees of freedom
	dofs_inlet_x = dfx.fem.locate_dofs_geometrical((sub_space_x, sub_space_x_collapsed), inlet)
	bcs_inlet_x = dfx.fem.dirichletbc(u_inlet_x, dofs_inlet_x, sub_space_x) # Same as OpenFOAM

	# Actual BCs
	spy.applyBCs(dofs_inlet_x[0],bcs_inlet_x) # x=X entirely handled by implicit Neumann

	# Same for tangential
	sub_space_th=spy.FS.sub(0).sub(2)
	sub_space_th_collapsed,_=sub_space_th.collapse()
	u_inlet_th=Function(sub_space_th_collapsed)
	u_inlet_th.interpolate(lambda x: S*x[1]*np.tanh(6*(1-x[1]**2))*(x[1]<1))
	dofs_inlet_th = dfx.fem.locate_dofs_geometrical((sub_space_th, sub_space_th_collapsed), inlet)
	bcs_inlet_th = dfx.fem.dirichletbc(u_inlet_th, dofs_inlet_th, sub_space_th) # Same as OpenFOAM
	spy.applyBCs(dofs_inlet_th[0],bcs_inlet_th)

	# Handle homogeneous boundary conditions
	spy.applyHomogeneousBCs([(inlet,['r']),(nozzle,['x','r','th']),(symmetry,['r','th'])])
	spy.applyHomogeneousBCs([(nozzle,['x','r','th'])],2) # Enforce nu=0 at the wall
	# Have small but !=0 nu at inlet
	for direction in ['x','r','th']:
		dofs,bcs=spy.constantBC(direction,inlet,3/Re,2)
		spy.applyBCs(dofs,bcs)

# Baseflow (really only need DirichletBC objects) enforces :
# u=0 at inlet, nozzle & top (linearise as baseflow)
# u_r, u_th=0 for symmetry axis if m=0, u_x=0 if |m|=1, u=0 else (derived from momentum csv r th as r->0)
# However there are hidden BCs in the weak form !
# Because of the IPP, we have nu grad u.n=p n everywhere. This gives :
# d_ru_x=0 for symmetry axis (if not overwritten by above)
# d_xu_x/Re=p, d_xu_r=0, d_xu_th=0 at outflow (free flow)
def boundaryConditionsPerturbations(spy:SPY,m:int) -> None:
	# Handle homogeneous boundary conditions
	homogeneous_boundaries=[(inlet,['x','r','th']),(nozzle,['x','r','th'])]
	if 	     m ==0: homogeneous_boundaries.append((symmetry,['r','th']))
	elif abs(m)==1: homogeneous_boundaries.append((symmetry,['x']))
	else:		    homogeneous_boundaries.append((symmetry,['x','r','th']))
	spy.applyHomogeneousBCs(homogeneous_boundaries)

def weakBoundaryConditions(spy:SPY,u,_,m:int=0) -> ufl.Form:
	boundaries = [(2, nozzle)]

	facet_indices, facet_markers = [], []
	fdim = spy.mesh.topology.dim - 1
	for (marker, locator) in boundaries:
		facets = dfx.mesh.locate_entities(spy.mesh, fdim, locator)
		facet_indices.append(facets)
		facet_markers.append(np.full_like(facets, marker))
	
	facet_indices = np.hstack(facet_indices).astype(np.int32)
	facet_markers = np.hstack(facet_markers).astype(np.int32)
	sorted_facets = np.argsort(facet_indices)
	
	face_tag = dfx.mesh.meshtags(spy.mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
	ds = ufl.Measure("ds", domain=spy.mesh, subdomain_data=face_tag)
	n = ufl.FacetNormal(spy.mesh)
	n = ufl.as_vector([n[0],n[1],0])

	v,_,_=ufl.split(spy.test)

	grd,r=lambda v: spy.grd(v,m),spy.r
	nu=1/spy.Re+spy.nut
	
	weak_bcs=-ufl.inner(nu* grd(u).T*n,    v)     *ds(1)
	weak_bcs-=ufl.inner(nu*(grd(u).T*n)[0],v[0])*r*ds(2)

	return weak_bcs
