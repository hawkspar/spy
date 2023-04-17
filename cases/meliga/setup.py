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

# Geometry parameters (validation legacy)
x_max=120; r_max=60
x_phy=70;  r_phy=10

# Physical Parameters
Re=200
Re_s=.1

# Numerical parameters
params = {"rp":.99,    #relaxation_parameter
		  "atol":1e-6, #absolute_tolerance
		  "rtol":1e-9, #DOLFIN_EPS does not work well
		  "max_iter":100}
datapath='meliga' #folder for results
direction_map={'x':0,'r':1,'th':2}

# Geometry
def inlet( x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[0],0,	params['atol']) # Left border
def outlet(x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[0],x_max,params['atol']) # Right border
def top(   x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[1],r_max,params['atol']) # Top boundary at r=R


# Sponged Reynolds number
def Ref(spy:SPY) -> dfx.Function:
	# Damped Reynolds number
	def csi(a,b,l): return .5*(1+np.tanh(4*np.tan(-np.pi/2+np.pi*np.abs(a-b)/l)))
	def sponged_Reynolds(x):
		Rem=np.ones(x[0].size)*Re
		x_ext=x[0]>x_phy
		Rem[x_ext]=Re		 +(Re_s-Re) 	   *csi(np.minimum(x[0][x_ext],x_max),x_phy, x_max-x_phy) # min necessary to prevent spurious jumps because of mesh conversion
		r_ext=x[1]>r_phy
		Rem[r_ext]=Rem[r_ext]+(Re_s-Rem[r_ext])*csi(np.minimum(x[1][r_ext],r_max),r_phy, r_max-r_phy)
		return Rem
	Red=dfx.Function(dfx.FunctionSpace(spy.mesh,ufl.FiniteElement("Lagrange",spy.mesh.ufl_cell(),2)))
	Red.interpolate(lambda x: sponged_Reynolds(x))
	return Red

# No turbulent visosity for this case
def nutf(spy:SPY,S:float): spy.nut=0

# Grabovski-Berger vortex with final slope
def grabovski_berger(r) -> np.ndarray:
	psi=(r_max-r)/(r_max-r_phy)/r_phy
	mr=r<1
	psi[mr]=r[mr]*(2-r[mr]**2)
	ir=(r>=1)*(r<r_phy)
	psi[ir]=1/r[ir]
	return psi

class InletAzimuthalVelocity():
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
def boundaryConditionsBaseflow(spyb:SPYB) -> None:	
	# Compute DoFs
	sub_space_th=spyb.TH.sub(0).sub(2)
	sub_space_th_collapsed=sub_space_th.collapse()
	
	# Degrees of freedom
	dofs_inlet_th = dfx.fem.locate_dofs_geometrical((sub_space_th, sub_space_th_collapsed), inlet)
	bcs_inlet_th = dfx.DirichletBC(spyb.u_inlet_th, dofs_inlet_th, sub_space_th) # u_th=S*psi(r) at x=0

	# Actual BCs
	dofs_inlet_x, bcs_inlet_x = spyb.constantBC('x',inlet,1) # u_x =1

	spyb.applyBCs(np.union1d(dofs_inlet_th,dofs_inlet_x),
							 [bcs_inlet_th, bcs_inlet_x]) # x=X entirely handled by implicit Neumann
	
	# Handle homogeneous boundary conditions
	spyb.applyHomogeneousBCs([(inlet,['r']),(top,['r','th']),(spyb.symmetry,['r','th'])])

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
	if 	     m ==0: homogeneous_boundaries.append((spy.symmetry,['r','th']))
	#elif abs(m)==1: homogeneous_boundaries.append((spy.symmetry,['x']))
	else:		    homogeneous_boundaries.append((spy.symmetry,['x','r','th']))
	spy.applyHomogeneousBCs(homogeneous_boundaries)