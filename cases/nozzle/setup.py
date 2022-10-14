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
S,Re=0,400000
hb,hp=5e-10,7e-8

# Numerical Parameters
params = {"rp":.95,    #relaxation_parameter
		  "atol":1e-9, #absolute_tolerance
		  "rtol":1e-6, #DOLFIN_EPS does not work well
		  "max_iter":100}
datapath='nozzle/' #folder for results
direction_map={'x':0,'r':1,'th':2}

# Geometry
def inlet( x:ufl.SpatialCoordinate)   -> np.ndarray: return np.isclose(x[0],0,		   params['atol']) # Left border
def outlet(x:ufl.SpatialCoordinate)   -> np.ndarray: return np.isclose(x[0],np.max(x[0]),params['atol']) # Right border
def top(   x:ufl.SpatialCoordinate)   -> np.ndarray: return np.isclose(x[1],np.max(x[1]),params['atol']) # Top (tilded) boundary
def nozzle(x:ufl.SpatialCoordinate,h) -> np.ndarray: return (x[1]<R+h+params['atol'])*(R-params['atol']<x[1])*(x[0]<R+params['atol'])

def Ref(spy:SPY): return Re

#def forcingIndicator(x): return np.isclose(x[1],R,.2)*(x[0]<R+.2)

def nutf(spy:SPY,S,Re):
	loadStuff(spy.nut_path,['S','Re'],[S,Re],spy.nut)
	spy.nut.x.array[spy.nut.x.array<0]=0 # Enforce positive

class InletAzimuthalVelocity():
	def __init__(self, S): self.S = 0
	def __call__(self, x): return self.S*x[0]

def boundaryConditionsBaseflow(spy:SPY) -> None:
	# Compute DoFs
	sub_space_x=spy.TH.sub(0).sub(0)
	sub_space_x_collapsed,_=sub_space_x.collapse()

	u_inlet_x=Function(sub_space_x_collapsed)
	u_inlet_x.interpolate(lambda x: np.tanh(6*(1-x[1]**2))*(x[1]<1)+
							    .05*np.tanh(6*(x[1]**2-1))*(x[1]>1))
	
	# Degrees of freedom
	dofs_inlet_x = dfx.fem.locate_dofs_geometrical((sub_space_x, sub_space_x_collapsed), inlet)
	bcs_inlet_x = dfx.fem.dirichletbc(u_inlet_x, dofs_inlet_x, sub_space_x) # Same as OpenFOAM

	# Actual BCs
	spy.applyBCs(dofs_inlet_x[0],bcs_inlet_x) # x=X entirely handled by implicit Neumann

	# Same for tangential
	sub_space_th=spy.TH.sub(0).sub(2)
	sub_space_th_collapsed,_=sub_space_th.collapse()

	# Modified vortex that goes to zero at top boundary
	u_inlet_th=Function(sub_space_th_collapsed)
	spy.inlet_azimuthal_velocity=InletAzimuthalVelocity(0)
	u_inlet_th.interpolate(spy.inlet_azimuthal_velocity)
	dofs_inlet_th = dfx.fem.locate_dofs_geometrical((sub_space_th, sub_space_th_collapsed), inlet)
	bcs_inlet_th = dfx.fem.dirichletbc(u_inlet_th, dofs_inlet_th, sub_space_th) # Same as OpenFOAM
	spy.applyBCs(dofs_inlet_th[0],bcs_inlet_th)

	# Handle homogeneous boundary conditions
	spy.applyHomogeneousBCs([(lambda x: nozzle(x,hb),['x','r','th']),(spy.symmetry,['r','th'])])

# Baseflow (really only need DirichletBC objects) enforces :
# u=0 at inlet, nozzle & top (linearise as baseflow)
# u_r, u_th=0 for symmetry axis if m=0, u_x=0 if |m|=1, u=0 else (derived from momentum csv r th as r->0)
# However there are hidden BCs in the weak form !
# Because of the IPP, we have nu grad u.n=p n everywhere. This gives :
# d_ru_x=0 for symmetry axis (if not overwritten by above)
# d_xu_x/Re=p, d_xu_r=0, d_xu_th=0 at outflow (free flow)
def boundaryConditionsPerturbations(spy:SPY,m:int) -> None:
	# Handle homogeneous boundary conditions
	homogeneous_boundaries=[(inlet,['x','r','th']),(lambda x: nozzle(x,hp),['x','r','th']),(top,['x','r','th'])]
	if 	     m ==0: homogeneous_boundaries.append((spy.symmetry,['r','th']))
	elif abs(m)==1: homogeneous_boundaries.append((spy.symmetry,['x']))
	else:		    homogeneous_boundaries.append((spy.symmetry,['x','r','th']))
	spy.applyHomogeneousBCs(homogeneous_boundaries)

def weakBoundaryConditions(spy:SPY,u,p,m:int=0) -> ufl.Form:
	boundaries = [(1, lambda x: np.logical_or(top(x),outlet(x))), (2, spy.symmetry), (3, lambda x: nozzle(x,hp))]

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

	v,s=ufl.split(spy.test)

	grd=lambda v: spy.grd(v,m)
	nu=1/spy.Re+spy.nut
	
	weak_bcs=-ufl.inner(nu* grd(u).T*n,    	   v)   *ds(1)
	weak_bcs-=ufl.inner(nu*(grd(u).T*n)[0],	   v[0])*ds(2)
	weak_bcs+=ufl.inner(	grd(p),		   	   s*n) *ds(3)

	return weak_bcs
