# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import re
import numpy as np
import dolfinx as dfx
import os, ufl, shutil, re
from petsc4py import PETSc as pet
from mpi4py.MPI import COMM_WORLD as comm
from dolfinx.fem import FunctionSpace, Function

p0=comm.rank==0

def dirCreator(path:str):
	if not os.path.isdir(path):
		comm.barrier() # Wait for all other processors
		if p0: os.mkdir(path)
		return True

# Simple handler
def meshConvert(path:str,cell_type:str) -> None:
	if p0:
		import meshio #pip3 install --no-binary=h5py h5py meshio
		gmsh_mesh = meshio.read(path+".msh")
		# Write it out again
		ps = gmsh_mesh.points[:,:2]
		cs = gmsh_mesh.get_cells_type(cell_type)
		dolfinx_mesh = meshio.Mesh(points=ps, cells={cell_type: cs})
		meshio.write(path+".xdmf", dolfinx_mesh)
	comm.barrier()

# Memoisation routine - find closest in param
def loadStuff(params:list,path:str,keys:list,vector:pet.Vec) -> None:
	closest_file_name=path
	if dirCreator(path): return
	file_names = [f for f in os.listdir(path) if f[-3:]=="dat"]
	d=np.infty
	for file_name in file_names:
		match = re.search(r'_n=(\d*)',file_name)
		n = int(match.group(1))
		if n!=comm.size: continue # Don't read if != proc nb
		fd=0 # Compute distance according to all params
		for param,key in zip(params,keys):
			match = re.search(r'_'+key+r'=(\d*\.?\d*)',file_name)
			param_file = float(match.group(1)) # Take advantage of file format
			fd += abs(param-param_file)
		if fd<d: d,closest_file_name=fd,path+file_name
	try:
		viewer = pet.Viewer().createMPIIO(closest_file_name, 'r', comm)
		vector.load(viewer)
		vector.ghostUpdate(addv=pet.InsertMode.INSERT, mode=pet.ScatterMode.FORWARD)
		# Loading eddy viscosity too
		if p0: print("Loaded "+closest_file_name)
	except:
		if p0: print("Error loading file ! Moving on...")

# Swirling Parallel Yaj
class SPY:
	def __init__(self, params:dict, datapath:str, Ref, nutf, direction_map:dict, forcingIndicator=None) -> None:
		# Direction dependant
		self.direction_map=direction_map
		# Solver parameters (Newton mostly, but also eig)
		self.params=params

		# Paths
		self.case_path		 ='/home/shared/cases/'+datapath
		self.baseflow_path   =self.case_path+'baseflow/'
		self.nut_path		 =self.baseflow_path+'nut/'
		self.dat_real_path	 =self.baseflow_path+'dat_real/'
		self.dat_complex_path=self.baseflow_path+'dat_complex/'
		self.print_path		 =self.baseflow_path+'print/'
		self.npy_path		 =self.baseflow_path+'npy/'
		self.resolvent_path	 =self.case_path+'resolvent/'
		self.eig_path		 =self.case_path+'eigenvalues/'

		# Mesh from file
		meshpath=self.case_path+datapath[:-1]+".xdmf"
		with dfx.io.XDMFFile(comm, meshpath, "r") as file:
			self.mesh = file.read_mesh(name="Grid")

		# Extraction of r
		self.r = ufl.SpatialCoordinate(self.mesh)[direction_map['r']]
		# Local cell number
		tdim = self.mesh.topology.dim
		num_cells = self.mesh.topology.index_map(tdim).size_local + self.mesh.topology.index_map(tdim).num_ghosts
	
		# Finite elements & function spaces
		FE_vector  =ufl.VectorElement("CG",self.mesh.ufl_cell(),2,3)
		FE_scalar  =ufl.FiniteElement("CG",self.mesh.ufl_cell(),1)
		FE_constant=ufl.FiniteElement("DG",self.mesh.ufl_cell(),0)
		V = FunctionSpace(self.mesh,FE_scalar)
		W = FunctionSpace(self.mesh,FE_constant)
		self.TH=FunctionSpace(self.mesh,FE_vector*FE_scalar) # Taylor Hodd elements ; stable element pair
		self.u_space, self.u_dofs = self.TH.sub(0).collapse()
		self.u_dofs=np.array(self.u_dofs)
		
		# Test & trial functions
		self.trial = ufl.TrialFunction(self.TH)
		self.test  = ufl.TestFunction( self.TH)
		
		# Re computation
		self.Re = Ref(self)
		# Initialisation of q
		self.q = Function(self.TH)
		# Turbulent viscosity
		self.nut  = Function(V)
		self.nutf = lambda S: nutf(self,S)
		# Local mesh size
		self.h = Function(W)
		self.h.x.array[:]=dfx.cpp.mesh.h(self.mesh, tdim, range(num_cells))

		# Forcing localisation
		if forcingIndicator==None: self.indic=1
		else:
			self.indic = Function(V)
			self.indic.interpolate(forcingIndicator)

		# BCs essentials
		self.dofs = np.empty(0,dtype=np.int32)
		self.bcs=[]

	# Jet geometry
	def symmetry(self, x:ufl.SpatialCoordinate) -> np.ndarray:
		return np.isclose(x[self.direction_map['r']],0,self.params['atol']) # Axis of symmetry at r=0

	# Gradient with r multiplication
	def rgrd(self,v,m):
		r=self.r
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		if isinstance(v,ufl.indexed.Indexed):
			return ufl.as_vector([r*v.dx(dx), r*v.dx(dr), m*1j*v])
		return ufl.as_tensor([[r*v[dx].dx(dx), r*v[dx].dx(dr), m*1j*v[dx]],
							  [r*v[dr].dx(dx), r*v[dr].dx(dr), m*1j*v[dr]-v[dt]],
							  [r*v[dt].dx(dx), r*v[dt].dx(dr), m*1j*v[dt]+v[dr]]])

	def grdr(self,v,m):
		r=self.r
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		return ufl.as_tensor([[r*v[dx].dx(dx), v[dx]+r*v[dx].dx(dr), m*1j*v[dx]],
					  		  [r*v[dr].dx(dx), v[dr]+r*v[dr].dx(dr), m*1j*v[dr]-v[dt]],
							  [r*v[dt].dx(dx), v[dt]+r*v[dt].dx(dr), m*1j*v[dt]+v[dr]]])

	def rdiv(self,v,m):
		r=self.r
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		return r*v[dx].dx(dx) +   v[dr] + r*v[dr].dx(dr) + m*1j*v[dt]

	def divr(self,v,m):
		r=self.r
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		return r*v[dx].dx(dx) + 2*v[dr] + r*v[dr].dx(dr) + m*1j*v[dt]
	
	def r2vis(self,v,m):
		r=self.r
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		nu=1/self.Re+self.nut
		return ufl.as_vector([2*r**2*(nu*v[dx].dx(dx)).dx(dx)			  +r*(r*nu*(v[dr].dx(dx)+v[dx].dx(dr))).dx(dr)+m*1j*nu*(r*v[dt].dx(dx)+m*1j*v[dx]),
							  r**2*(nu*(v[dr].dx(dx)+v[dx].dx(dr))).dx(dx)+r*(2*r*nu*v[dr].dx(dr)).dx(dr)			  +m*1j*nu*(r*v[dt].dx(dr)+m*1j*v[dr])-2*m*1j*nu*v[dt],
							  r  *(nu*(r*v[dt].dx(dx)+m*1j*v[dx])).dx(dx) +r*(nu*(r*v[dt].dx(dr)+m*1j*v[dr])).dx(dr)-2*m**2*nu*v[dt]					  +nu*(r*v[dt].dx(dr)+m*1j*v[dr])])

	# Helper
	def loadBaseflow(self,S,Re,m):
		# Load baseflow
		loadStuff([S,Re],self.dat_complex_path,['S','Re'],self.q.vector)
		# Load turbulent viscosity
		self.nutf(S)

		# Split Arguments
		v,_ = ufl.split(self.test)
		U,_ = ufl.split(self.q) # Baseflow

		h,nu=self.h,1/self.Re+self.nut

		# Weird Johann local tau (SUPG stabilisation)
		n=ufl.sqrt(ufl.inner(U,U))
		Pe=ufl.real(n*h/nu)
		cPe=ufl.conditional(ufl.le(Pe,3),Pe/3,1)
		t=cPe*h/2/n

		self.SUPG = t*self.rgrd(v,m)*U # Streamline Upwind Petrov Galerkin
		self.SUPG *= 0
	
	# Heart of this entire code
	def navierStokes(self) -> ufl.Form:
		# Shortforms
		r,nu=self.r,1/self.Re+self.nut
		rdiv,divr,rgrd,grdr=self.rdiv,self.divr,self.rgrd,self.grdr
		
		# Functions
		u,p=ufl.split(self.q)
		v,s=ufl.split(self.test)
		
		# Mass (variational formulation)
		F  = ufl.inner(	   rdiv(u,0),     r**2*s)
		# Momentum (different test functions and IBP)
		F += ufl.inner(    rgrd(u,0)*u,   r**2*v) # Convection
		F -= ufl.inner(	   r*   p,	    divr(r*v,0)) # Pressure
		F += ufl.inner(nu*(rgrd(u,0)+
					   	   rgrd(u,0).T),grdr(r*v,0)) # Diffusion (grad u.T significant with nut)
		return F*ufl.dx
		
	# Not automatic because of convection term
	def linearisedNavierStokes(self,m:int) -> ufl.Form:
		# Shortforms
		r,nu=self.r,1/self.Re+self.nut
		rdiv,divr,rgrd,grdr=self.rdiv,self.divr,self.rgrd,self.grdr
		SUPG,r2vis=self.SUPG,self.r2vis
		
		# Functions
		u, p=ufl.split(self.trial)
		U, _=ufl.split(self.q) # Baseflow
		v, s=ufl.split(self.test)
		
		# Mass (variational formulation)
		F  = ufl.inner(	   rdiv( u,m),     r**2*s)
		# Momentum (different test functions and IBP)
		F += ufl.inner(	   rgrd( U,0)*u,   r**2*v+r*SUPG) # Convection
		F += ufl.inner(	   rgrd( u,m)*U,   r**2*v+r*SUPG)
		F -= ufl.inner(	   r*	 p,      divr(r*v,m)) # Pressure
		F += ufl.inner(	   rgrd( p,m),	 	      r*SUPG)
		F += ufl.inner(nu*(rgrd( u,m)+
					   	   rgrd( u,m).T),grdr(r*v,m)) # Diffusion (grad u.T significant with nut)
		F -= ufl.inner(	   r2vis(u,m),			    SUPG)
		return F*ufl.dx
		
	# Code factorisation
	def constantBC(self, direction:chr, boundary, value=0) -> tuple:
		sub_space=self.TH.sub(0).sub(self.direction_map[direction])
		sub_space_collapsed,_=sub_space.collapse()
		# Compute unflattened DoFs (don't care for flattened ones)
		dofs,_ = dfx.fem.locate_dofs_geometrical((sub_space, sub_space_collapsed), boundary)
		# Actual BCs
		bcs = dfx.fem.dirichletbc(pet.ScalarType(value), dofs, sub_space) # u_i=value at boundary
		return dofs,bcs

	# Encapsulation	
	def applyBCs(self, dofs:np.ndarray,bcs:list):
		self.dofs=np.union1d(dofs,self.dofs)
		self.bcs.extend(bcs)

	def applyHomogeneousBCs(self, tup:list):
		for marker,directions in tup:
			for direction in directions:
				dofs,bcs=self.constantBC(direction,marker)
				self.applyBCs(dofs,[bcs])

	def saveStuff(self,dir:str,name:str,fun:Function) -> None:
		dirCreator(dir)
		with dfx.io.XDMFFile(comm, dir+name+".xdmf", "w") as xdmf:
			xdmf.write_mesh(self.mesh)
			xdmf.write_function(fun)
	
	def saveStuffMPI(self,dir:str,name:str,vec:pet.Vec) -> None:
		dirCreator(dir)
		viewer = pet.Viewer().createMPIIO(dir+name+f"_n={comm.size:d}.dat", 'w', comm)
		vec.view(viewer)

	# Converters
	def datToNpy(self,fi:str,fo:str) -> None:
		viewer = pet.Viewer().createMPIIO(fi, 'r', comm)
		self.q.vector.load(viewer)
		self.q.x.scatter_forward()
		np.save(fo,self.q.x.array)

	def datToNpyAll(self) -> None:
		if os.path.isdir(self.dat_real_path):
			file_names = [f for f in os.listdir(self.dat_real_path) if f[-3:]=="dat"]
			dirCreator(self.npy_path)
			for file_name in file_names:
				self.datToNpy(self.dat_real_path+file_name,
							self.npy_path+file_name[:-4]+f"_p={comm.rank:d}.npy")
			if p0: shutil.rmtree(self.dat_real_path)

	def npyToDat(self,fi:str,fo:str) -> None:
		self.q.x.array.real=np.load(fi,allow_pickle=True)
		self.q.x.scatter_forward()
		viewer = pet.Viewer().createMPIIO(fo, 'w', comm)
		self.q.vector.view(viewer)
	
	def npyToDatAll(self) -> None:
		if os.path.isdir(self.npy_path):
			file_names = [f for f in os.listdir(self.npy_path)]
			dirCreator(self.dat_complex_path)
			for file_name in file_names:
				match = re.search(r'_p=([0-9]*)',file_name)
				if comm.rank==int(match.group(1)):
					self.npyToDat(self.npy_path+file_name,
								self.dat_complex_path+file_name[:-8]+".dat")
			if p0: shutil.rmtree(self.npy_path)
	
	# Quick check functions
	def sanityCheckU(self):
		u,_=self.q.split()
		self.saveStuff("./","sanity_check_u",u)

	def sanityCheck(self):
		u,p=self.q.split()
		self.saveStuff("./","sanity_check_u",u)
		self.saveStuff("./","sanity_check_p",p)
		self.saveStuff("./","sanity_check_nut",self.nut)

	def sanityCheckBCs(self):
		v=Function(self.TH)
		v.vector.zeroEntries()
		v.x.array[self.dofs]=np.ones(self.dofs.size)
		u,_=v.split()
		self.saveStuff("","sanity_check_bcs",u)