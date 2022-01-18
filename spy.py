# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import numpy as np
import dolfinx as dfx
import os, ufl, shutil
from dolfinx.io import XDMFFile
from petsc4py import PETSc as pet
from mpi4py.MPI import COMM_WORLD

# Swirling Parallel Yaj
class spy:
	def __init__(self,meshpath:str,datapath:str,dM:float) -> None:
		# TBI : problem dependent
		self.direction_map={'x':0,'r':1,'th':2}

		# Geometry parameters
		self.x_max=30; self.r_max=10
		self.x_phy=20; self.r_phy=5

		# Solver parameters
		self.rp  =.99 #relaxation_parameter
		self.atol=1e-6 #absolute_tolerance
		self.rtol=1e-9 #DOLFIN_EPS does not work well
		self.max_iter=100

		# Paths
		if not os.path.isdir(datapath): os.mkdir(datapath)
		self.dat_real_path	 =datapath+'baseflow/dat_real/'
		self.dat_complex_path=datapath+'baseflow/dat_complex/'
		self.npy_path		 =datapath+'baseflow/npy/'
		self.resolvent_path	 =datapath+'/resolvent/'
		self.eig_path		 =datapath+'/eigenvalues/'

		# Mesh from file
		with XDMFFile(COMM_WORLD, meshpath, "r") as file:
			self.mesh = file.read_mesh(name="Grid")
		
		# Taylor Hodd elements ; stable element pair
		FE_vector=ufl.VectorElement("Lagrange",self.mesh.ufl_cell(),2,3)
		FE_scalar=ufl.FiniteElement("Lagrange",self.mesh.ufl_cell(),1)
		self.Space=dfx.FunctionSpace(self.mesh,FE_vector*FE_scalar)	# full vector function space
		
		# Test & trial functions
		self.Test  = ufl.TestFunction(self.Space)
		self.Trial = ufl.TrialFunction(self.Space)
		
		# Extraction of r and Re computation
		self.r = ufl.SpatialCoordinate(self.mesh)[1]
		self.d = self.dampingFactor(dM)
		with XDMFFile(COMM_WORLD, self.datapath+"d.xdmf", "w") as xdmf:
			xdmf.write_mesh(self.mesh)
			xdmf.write_function(self.d)
		self.q = dfx.Function(self.Space) # Initialisation of q

	# Jet geometry
	def inlet(	 self, x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[0],0,	  	 self.atol) # Left border
	def symmetry(self, x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[1],0,	  	 self.atol) # Axis of symmetry at r=0
	def outlet(	 self, x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[0],self.x_max,self.atol) # Right border
	def top(	 self, x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[1],self.r_max,self.atol) # Top boundary at r=R
	def nozzle(	 self, x:ufl.SpatialCoordinate) -> np.ndarray: return np.logical_and(x[0]<1,np.isclose(x[1],1,self.atol)) # Nozzle

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

	def csi(self,a,b,l): return .5*(1+np.tanh(4*np.tan(-np.pi/2+np.pi*np.abs(a-b)/l)))

	def df(self,x,dM):
		dm=np.zeros(x[0].size)
		x_ext=x[0]>self.x_phy
		dm[x_ext]= 	   		 dM			  *self.csi(np.minimum(x[0][x_ext],self.x_max),self.x_phy, self.x_max-self.x_phy) # min necessary to prevent spurious jumps because of mesh conversion
		r_ext=x[1]>self.r_phy
		dm[r_ext]=dm[r_ext]+(dM-dm[r_ext])*self.csi(np.minimum(x[1][r_ext],self.r_max),self.r_phy, self.r_max-self.r_phy)
		return np.maximum(dm,0)

	# Sponged Reynolds number
	def dampingFactor(self,dM) -> dfx.Function:
		d=dfx.Function(dfx.FunctionSpace(self.mesh,ufl.FiniteElement("Lagrange",self.mesh.ufl_cell(),2)))
		d.interpolate(lambda x: self.df(x,dM))
		return d
		
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

	def NavierStokes(self,Re:float) -> ufl.Form:
		# Shortforms
		r,d=self.r,self.d
		rdiv,divr,rgrad,gradr=self.rdiv,self.divr,self.rgrad,self.gradr
		u,p=ufl.split(self.q)
		v,w=ufl.split(self.Test)
		
		# Mass (variational formulation)
		F  = ufl.inner( rdiv(u,0), 	   	 w)
		# Momentum (different test functions and IBP)
		F += ufl.inner(rgrad(u,0)*u,   r*v)       # Convection
		F += ufl.inner(rgrad(u,0), gradr(v,0))/Re # Diffusion
		F -= ufl.inner(r*p,		 	divr(v,0)) 	  # Pressure
		# Numerical damping
		F -= ufl.inner(r*u,r*v)*d
		return F*ufl.dx
		
	# Not automatic because of convection term
	def LinearisedNavierStokes(self,Re:float,m:int) -> ufl.Form:
		# Shortforms
		r,d=self.r,self.d
		rdiv,divr,rgrad,gradr=self.rdiv,self.divr,self.rgrad,self.gradr
		u,	p=ufl.split(self.Trial)
		u_b,_=ufl.split(self.q) # Baseflow
		v,	w=ufl.split(self.Test)
		
		# Mass (variational formulation)
		F  = ufl.inner( rdiv(u,  m), 	   w)
		# Momentum (different test functions and IBP)
		F += ufl.inner(rgrad(u_b,0)*u,   r*v)    	# Convection
		F += ufl.inner(rgrad(u,  m)*u_b, r*v)
		F += ufl.inner(rgrad(u,  m), gradr(v,m))/Re # Diffusion
		F -= ufl.inner(r*p,			  divr(v,m)) 	# Pressure
		# Numerical damping
		F -= ufl.inner(r*u,r*v)*d
		return F*ufl.dx

	# Converters
	def datToNpy(self,fi,fo) -> None:
		viewer = pet.Viewer().createMPIIO(fi, 'r', COMM_WORLD)
		self.q.vector.load(viewer)
		self.q.vector.ghostUpdate(addv=pet.InsertMode.INSERT, mode=pet.ScatterMode.FORWARD)
		np.save(fo,self.q.x.array)

	def datToNpyAll(self) -> None:
		file_names = [f for f in os.listdir(self.dat_real_path) if f[-3]=="dat"]
		if not os.path.isdir(self.npy_path): os.mkdir(self.npy_path)
		for file_name in file_names:
			self.datToNpy(self.dat_real_path+file_name,
						  self.npy_path+file_name[:-3]+'npy')
		shutil.rmtree(self.dat_real_path)
		self.datToNpy(self.datapath+'last_baseflow_real.dat',self.datapath+'last_baseflow.npy')

	def npyToDat(self,fi,fo) -> None:
		self.q.vector.array.real=np.load(fi,allow_pickle=True)
		self.q.vector.scatter_forward()
		viewer = pet.Viewer().createMPIIO(fo, 'w', COMM_WORLD)
		self.q.x.view(viewer)
	
	def npyToDatAll(self) -> None:
		file_names = [f for f in os.listdir(self.npy_path)]
		if not os.path.isdir(self.dat_complex_path): os.mkdir(self.dat_complex_path)
		for file_name in file_names:
			self.npyToDat(self.npy_path+file_name,
						  self.dat_complex_path+file_name[:-3]+'dat')
		shutil.rmtree(self.npy_path)
		self.npyToDat(self.datapath+'last_baseflow.npy',self.datapath+'last_baseflow_complex.dat')