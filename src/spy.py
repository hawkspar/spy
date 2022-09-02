# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import re
import numpy as np
import dolfinx as dfx
import os, ufl, shutil, re
from dolfinx.io import XDMFFile
from petsc4py import PETSc as pet
from mpi4py.MPI import COMM_WORLD as comm

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
		with XDMFFile(comm, meshpath, "r") as file:
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
		V = dfx.FunctionSpace(self.mesh,FE_scalar)
		W = dfx.FunctionSpace(self.mesh,FE_constant)
		self.TH=dfx.FunctionSpace(self.mesh,FE_vector*FE_scalar) # Taylor Hodd elements ; stable element pair
		self.u_space, self.u_dofs = self.TH.sub(0).collapse(collapsed_dofs=True)
		self.u_dofs=np.array(self.u_dofs)
		
		# Test & trial functions
		self.trial = ufl.TrialFunction(self.TH)
		self.test  = ufl.TestFunction( self.TH)
		
		# Re computation
		self.Re = Ref(self)
		# Initialisation of q
		self.q = dfx.Function(self.TH)
		# Turbulent viscosity
		self.nut = dfx.Function(V)
		self.nutf = lambda S: nutf(self,S)
		# Local mesh size
		self.h = dfx.Function(W)
		self.h.x.array[:]=dfx.cpp.mesh.h(self.mesh, tdim, range(num_cells))

		# Forcing localisation
		if forcingIndicator==None: self.indic=1
		else:
			self.indic = dfx.Function(V)
			self.indic.interpolate(forcingIndicator)

		# BCs essentials
		self.dofs = np.empty(0,dtype=np.int32)
		self.bcs=[]

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

		self.SUPG = t*self.rgrad(v,m)*U # Streamline Upwind Petrov Galerkin

	# Jet geometry
	def symmetry(self, x:ufl.SpatialCoordinate) -> np.ndarray:
		return np.isclose(x[self.direction_map['r']],0,self.params['atol']) # Axis of symmetry at r=0

	# Gradient with r multiplication
	def grad(self,v,m):
		r=self.r
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		if isinstance(v,ufl.indexed.Indexed):
			return ufl.as_vector([v.dx(dx), v.dx(dr), m*1j*v/r])
		return ufl.as_tensor([[v[dx].dx(dx), v[dx].dx(dr),  m*1j*v[dx]		 /r],
							  [v[dr].dx(dx), v[dr].dx(dr), (m*1j*v[dr]-v[dt])/r],
							  [v[dt].dx(dx), v[dt].dx(dr), (m*1j*v[dt]+v[dr])/r]])

	# Gradient with r multiplication
	def rgrad(self,v,m):
		r=self.r
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		if isinstance(v,ufl.indexed.Indexed):
			return ufl.as_vector([r*v.dx(dx), r*v.dx(dr), m*1j*v])
		return ufl.as_tensor([[r*v[dx].dx(dx), r*v[dx].dx(dr), m*1j*v[dx]],
							  [r*v[dr].dx(dx), r*v[dr].dx(dr), m*1j*v[dr]-v[dt]],
							  [r*v[dt].dx(dx), r*v[dt].dx(dr), m*1j*v[dt]+v[dr]]])

	def gradr(self,v,m):
		r=self.r
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		return ufl.as_tensor([[r*v[dx].dx(dx), v[dx]+r*v[dx].dx(dr), m*1j*v[dx]],
					  		  [r*v[dr].dx(dx), v[dr]+r*v[dr].dx(dr), m*1j*v[dr]-v[dt]],
							  [r*v[dt].dx(dx), v[dt]+r*v[dt].dx(dr), m*1j*v[dt]+v[dr]]])

	def div(self,v,m):
		r=self.r
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		if len(v.ufl_shape)==1:
			return v[dx].dx(dx) + (r*v[dr]).dx(dr)/r + m*1j*v[dt]/r
		return ufl.as_vector([[v[dx,dx].dx(dx)+(r*v[dr,dx]).dx(dr)/r+m*1j*v[dt,dx]/r],
					  		  [v[dx,dr].dx(dx)+(r*v[dr,dr]).dx(dr)/r+m*1j*v[dt,dr]/r-v[dt,dt]/r],
							  [v[dx,dt].dx(dx)+(r*v[dr,dt]).dx(dr)/r+m*1j*v[dt,dt]+v[dt,dr]/r]])


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
	
	# Heart of this entire code
	def navierStokes(self) -> ufl.Form:
		# Shortforms
		r=self.r
		rdiv,divr,rgrad,gradr=self.rdiv,self.divr,self.rgrad,self.gradr
		u,p=ufl.split(self.q)
		v,s=ufl.split(self.test)
		
		# Mass (variational formulation)
		F  = ufl.inner(rdiv(u,0), 				   r*s)
		# Momentum (different test functions and IBP)
		F += ufl.inner(rgrad(u,0)*u, 			   r*v) # Convection
		F += ufl.inner(rgrad(u,0)+rgrad(u,0).T,gradr(v,0))*(1/self.Re+self.nut) # Diffusion (grad u.T significant with nut)
		F -= ufl.inner(r*p,						divr(v,0)) # Pressure
		return F*ufl.dx
		
	# Not automatic because of convection term
	def linearisedNavierStokes(self,m:int) -> ufl.Form:
		# Shortforms
		r=self.r
		rdiv,divr,rgrad,gradr=self.rdiv,self.divr,self.rgrad,self.gradr
		SUPG,r2vis=self.SUPG,self.r2vis
		nu=1/self.Re+self.nut
		u, p=ufl.split(self.trial)
		U, _=ufl.split(self.q) # Baseflow
		v, s=ufl.split(self.test)
		
		# Mass (variational formulation)
		F  = ufl.inner( rdiv(u,m),   r**2*s)
		# Momentum (different test functions and IBP)
		F += ufl.inner(rgrad(U,0)*u, r**2*v+r*SUPG) # Convection
		F += ufl.inner(rgrad(u,m)*U, r**2*v+r*SUPG)
		F -= ufl.inner(r*  	  p,   divr(r*v,m)) # Pressure
		F += ufl.inner(rgrad(p,m),		    r*SUPG)
		F += ufl.inner(rgrad(u,m)+rgrad(u,m).T,gradr(r*v,m))*nu # Diffusion (grad u.T significant with nut)
		F -= ufl.inner(r2vis(u,m),		  	  SUPG)
		return F*ufl.dx
		
	# Code factorisation
	def constantBC(self, direction:chr, boundary, value=0) -> tuple:
		sub_space=self.TH.sub(0).sub(self.direction_map[direction])
		sub_space_collapsed=sub_space.collapse()
		# Compute proper values
		constant=dfx.Function(sub_space_collapsed)
		with constant.vector.localForm() as zero_loc: zero_loc.set(value)
		# Compute unflattened DoFs (don't care for flattened ones)
		dofs = dfx.fem.locate_dofs_geometrical((sub_space, sub_space_collapsed), boundary)
		# Actual BCs
		bcs = dfx.DirichletBC(constant, dofs, sub_space) # u_i=value at boundary
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

	def saveStuff(self,dir:str,name:str,fun:dfx.Function) -> None:
		dirCreator(dir)
		with XDMFFile(comm, dir+name+".xdmf", "w") as xdmf:
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
		v=dfx.Function(self.TH)
		v.vector.zeroEntries()
		v.x.array[self.dofs]=np.ones(self.dofs.size)
		u,_=v.split()
		self.saveStuff("","sanity_check_bcs",u)