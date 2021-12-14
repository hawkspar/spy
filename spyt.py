# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import os, ufl
import numpy as np
import dolfinx as dfx
from dolfinx.io import XDMFFile
from petsc4py import PETSc as pet
from mpi4py.MPI import COMM_WORLD

# Swirling Parallel Yaj Thing
class spyt:
	def __init__(self,meshpath:str,datapath:str,Re:float) -> None:
		# TBI : problem dependent
		self.direction_map={'x':0,'r':1,'th':2}

		# Geometry parameters
		self.x_max=70; self.r_max=10; self.l=50

		# Solver parameters
		self.rp  =.99 #relaxation_parameter
		self.atol=1e-9 #absolute_tolerance
		self.rtol=1e-14 #DOLFIN_EPS does not work well
		self.max_iter=60

		# Paths
		self.datapath = datapath
		self.baseflow_path ='baseflow/'

		# Mesh from file
		with XDMFFile(COMM_WORLD, meshpath, "r") as file:
			self.mesh = file.read_mesh(name="Grid")
		x = self.mesh.geometry.x
		gdim = self.mesh.geometry.dim
		x=x[:,:gdim]
		# Force positive quadrant
		x[:,0] -= np.minimum(x[:,0],0)
		x[:,1] -= np.minimum(x[:,1],0)
		# Force bounded pts
		x[:,0] += np.minimum(self.x_max+self.l-x[:,0],0)
		x[:,1] += np.minimum(self.r_max+self.l-x[:,1],0)
		
		# Taylor Hodd elements ; stable element pair
		FE_vector=ufl.VectorElement("Lagrange",self.mesh.ufl_cell(),2,3)
		FE_scalar=ufl.FiniteElement("Lagrange",self.mesh.ufl_cell(),1)
		self.Space=dfx.FunctionSpace(self.mesh,FE_vector*FE_scalar)	# full vector function space
		
		# Test & trial functions
		self.Test  = ufl.TestFunction(self.Space)
		self.Trial = ufl.TrialFunction(self.Space)
		
		# Extraction of r and Re computation
		self.r = ufl.SpatialCoordinate(self.mesh)[1]
		self.Re = self.sponged_Reynolds(Re)
		self.mu = 1/self.Re
		self.q = dfx.Function(self.Space) # Initialisation of q
		"""
		with XDMFFile(COMM_WORLD, self.datapath+"Re.xdmf", "w") as xdmf:
			xdmf.write_mesh(self.mesh)
			xdmf.write_function(self.Re)
		"""

	# Jet geometry
	def inlet(	 self, x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[0],0,	  			self.atol) # Left border
	def symmetry(self, x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[1],0,	  			self.atol) # Axis of symmetry at r=0
	def outlet(	 self, x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[0],self.x_max+self.l,self.atol) # Right border
	def top(	 self, x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[1],self.r_max+self.l,self.atol) # Top boundary at r=R

	# Gradient with x[0] is x, x[1] is r, x[2] is theta
	def rgrad(self,v,m):
		return ufl.as_tensor([[self.r*v[0].dx(0), self.r*v[0].dx(1), m*1j*v[0]],
							  [self.r*v[1].dx(0), self.r*v[1].dx(1), m*1j*v[1]-v[2]],
							  [self.r*v[2].dx(0), self.r*v[2].dx(1), m*1j*v[2]+v[1]]])

	def gradr(self,v,m):
		return ufl.as_tensor([[self.r*v[0].dx(0), v[0]+self.r*v[0].dx(1), m*1j*v[0]],
					  		  [self.r*v[1].dx(0), v[1]+self.r*v[1].dx(1), m*1j*v[1]-v[2]],
							  [self.r*v[2].dx(0), v[2]+self.r*v[2].dx(1), m*1j*v[2]+v[1]]])

	# Same for divergent
	def rdiv(self,v,m): return self.r*v[0].dx(0) +   v[1] + self.r*v[1].dx(1) + m*1j*v[2]

	def divr(self,v,m): return self.r*v[0].dx(0) + 2*v[1] + self.r*v[1].dx(1) + m*1j*v[2]

	def csi(self,a,b): return .5*(1+np.tanh(4*np.tan(-np.pi/2+np.pi*np.abs(a-b)/self.l)))

	def Ref(self,x,Re_s,Re):
		Rem=np.ones(x[0].size)*Re
		x_ext=x[0]>self.x_max
		Rem[x_ext]=Re		 +(Re_s-Re) 	   *self.csi(x[0][x_ext],self.x_max) # min necessary to prevent spurious jumps because of mesh conversion
		r_ext=x[1]>self.r_max
		Rem[r_ext]=Rem[r_ext]+(Re_s-Rem[r_ext])*self.csi(x[1][r_ext],self.r_max)
		return Rem

	# Sponged Reynolds number
	def sponged_Reynolds(self,Re) -> dfx.Function:
		Re_s=.1
		Red=dfx.Function(dfx.FunctionSpace(self.mesh,ufl.FiniteElement("Lagrange",self.mesh.ufl_cell(),2)))
		Red.interpolate(lambda x: self.Ref(x,Re_s,Re))
		return Red
		
	# Code factorisation
	def ConstantBC(self, direction:chr, boundary, value=0) -> tuple:
		sub_space=self.Space.sub(0).sub(self.direction_map[direction])
		sub_space_collapsed=sub_space.collapse()
		# Compute proper zeros
		constant=dfx.Function(sub_space_collapsed)
		with constant.vector.localForm() as zero_loc: zero_loc.set(value)
		# Compute DoFs
		dofs = dfx.fem.locate_dofs_geometrical((sub_space, sub_space_collapsed), boundary)
		# Actual BCs
		bcs = dfx.DirichletBC(constant, dofs, sub_space) # u_i=value at boundary
		return dofs[0], bcs # Only return unflattened dofs
		
	# Not automatic because of convection term
	def LinearisedNavierStokes(self,m:int) -> ufl.Form:
		# Shortforms
		r,mu=self.r,self.mu
		rdiv,divr,rgrad,gradr=self.rdiv,self.divr,self.rgrad,self.gradr
		u,	p  =ufl.split(self.Trial)
		u_b,p_b=ufl.split(self.q) # Baseflow
		v,	w  =ufl.split(self.Test)
		
		# Mass (variational formulation)
		F  = ufl.inner( rdiv(u,  m), 	   w)
		# Momentum (different test functions and IBP)
		F += ufl.inner(rgrad(u_b,0)*u,   r*v)    		   # Convection
		F += ufl.inner(rgrad(u,  m)*u_b, r*v)
		F += ufl.inner(rgrad(u,  m), gradr(v,m))*mu # Diffusion
		F -= ufl.inner(r*p,			  divr(v,m)) 		   # Pressure
		return F*ufl.dx

	# Converters
	def datToNpy(self,fi,fo) -> None:
		viewer = pet.Viewer().createMPIIO(fi, 'r', COMM_WORLD)
		self.q.vector.load(viewer)
		self.q.vector.ghostUpdate(addv=pet.InsertMode.INSERT, mode=pet.ScatterMode.FORWARD)
		np.save(fo,self.q.x.array)

	def datToNpyAll(self) -> None:
		file_names = [f for f in os.listdir(self.datapath+self.baseflow_path+'dat_real/') if f[-3:]=="dat"]
		for file_name in file_names:
			self.datToNpy(self.datapath+self.baseflow_path+'dat_real/'+file_name,
						  self.datapath+self.baseflow_path+'npy/'+file_name[:-3]+'npy')
		self.datToNpy(self.datapath+'last_baseflow_real.dat',self.datapath+'last_baseflow.npy')

	def npyToDat(self,fi,fo) -> None:
		self.q.vector.array.real=np.load(fi,allow_pickle=True)
		self.q.vector.scatter_forward()
		viewer = pet.Viewer().createMPIIO(fo, 'w', COMM_WORLD)
		self.q.x.view(viewer)
	
	def npyToDatAll(self) -> None:
		file_names = [f for f in os.listdir(self.datapath+self.baseflow_path+'npy/') if f[-3:]=="npy"]
		for file_name in file_names:
			self.npyToDat(self.datapath+self.baseflow_path+'npy/'+file_name,
						  self.datapath+self.baseflow_path+'dat_complex/'+file_name[:-3]+'dat')
		self.npyToDat(self.datapath+'last_baseflow.npy',self.datapath+'last_baseflow_complex.dat')