# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import numpy as np
import PetscBinaryIO
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
def meshConvert(path:str,out:str,cell_type:str) -> None:
	import meshio #pip3 install --no-binary=h5py h5py meshio
	gmsh_mesh = meshio.read(path+".msh")
	# Write it out again
	ps = gmsh_mesh.points[:,:2]
	cs = gmsh_mesh.get_cells_type(cell_type)
	dolfinx_mesh = meshio.Mesh(points=ps, cells={cell_type: cs})
	meshio.write(out+".xdmf", dolfinx_mesh)
	print("Mesh "+path+".msh converted to "+out+".xdmf !",flush=True)

# Memoisation routine - find closest in param
def loadStuff(path:str,keys:list,params:list,vector:pet.Vec,io:PetscBinaryIO) -> None:
	closest_file_name=path
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

	proc_name = closest_file_name.split("_p=")
	proc_name = proc_name[0]+f"_p={comm.rank:d}"+proc_name[1][1:]
	
	input_vector = io.readBinaryFile(proc_name)[0]
	vector[...] = input_vector
	vector.ghostUpdate(addv=pet.InsertMode.INSERT, mode=pet.ScatterMode.FORWARD)
	# Loading eddy viscosity too
	if p0: print("Loaded "+proc_name,flush=True)

# Naive save with dir creation
def saveStuff(dir:str,name:str,vec:pet.Vec,io:PetscBinaryIO) -> None:
	dirCreator(dir)
	io_vec = vec.array_w.view(PetscBinaryIO.Vec)
	io.writeBinaryFile(dir+name+f"_n={comm.size:d}_p={comm.rank:d}.dat", [io_vec])

# Swirling Parallel Yaj
class SPY:
	def __init__(self, params:dict, datapath:str, Ref, nutf, direction_map:dict, C:bool, forcingIndicator=None) -> None:
		# Direction dependant
		self.direction_map=direction_map
		# Solver parameters (Newton mostly, but also eig)
		self.params=params

		# Paths
		self.case_path	   ='/home/shared/cases/'+datapath
		self.baseflow_path =self.case_path+'baseflow/'
		self.u_path	 	   =self.baseflow_path+'u/'
		self.p_path		   =self.baseflow_path+'p/'
		self.nut_path	   =self.baseflow_path+'nut/'
		self.print_path	   =self.baseflow_path+'print/'
		self.resolvent_path=self.case_path+'resolvent/'
		self.eig_path	   =self.case_path+'eigenvalues/'

		# Mesh from file
		meshpath=self.case_path+datapath[:-1]+".xdmf"
		with dfx.io.XDMFFile(comm, meshpath, "r") as file:
			self.mesh = file.read_mesh(name="Grid")
		if p0: print("Loaded "+meshpath,flush=True)

		# file handler and complex mode
		self.io = PetscBinaryIO.PetscBinaryIO(complexscalars=C)
		self.C=C

		# Extraction of r
		self.r = ufl.SpatialCoordinate(self.mesh)[direction_map['r']]
		# Local cell number
		tdim = self.mesh.topology.dim
		num_cells = self.mesh.topology.index_map(tdim).size_local + self.mesh.topology.index_map(tdim).num_ghosts
	
		# Finite elements & function spaces
		FE_vector  =ufl.VectorElement("CG",self.mesh.ufl_cell(),2,3)
		FE_scalar  =ufl.FiniteElement("CG",self.mesh.ufl_cell(),1)
		FE_constant=ufl.FiniteElement("DG",self.mesh.ufl_cell(),0)
		U = FunctionSpace(self.mesh,FE_vector)
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
		# Initialisation of baseflow
		self.U = Function(U)
		self.P = Function(V)
		# Turbulent viscosity
		self.nut  = Function(V)
		self.nutf = nutf
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
	def grd(self,v,m,i=0):
		r=self.r
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		if isinstance(v,ufl.indexed.Indexed):
			return ufl.as_vector([r*v.dx(dx), i*v+r*v.dx(dr), m*1j*v])
		return ufl.as_tensor([[r*v[dx].dx(dx), i*v[dx]+r*v[dx].dx(dr), m*1j*v[dx]],
							  [r*v[dr].dx(dx), i*v[dr]+r*v[dr].dx(dr), m*1j*v[dr]-v[dt]],
							  [r*v[dt].dx(dx), i*v[dt]+r*v[dt].dx(dr), m*1j*v[dt]+v[dr]]])

	# Gradient with r multiplication
	def grd_nor(self,v,m):
		r=self.r
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		if isinstance(v,ufl.indexed.Indexed):
			return ufl.as_vector([v.dx(dx), v.dx(dr), m*1j*v/r])
		return ufl.as_tensor([[v[dx].dx(dx), v[dx].dx(dr),  m*1j*v[dx]		 /r],
							  [v[dr].dx(dx), v[dr].dx(dr), (m*1j*v[dr]-v[dt])/r],
							  [v[dt].dx(dx), v[dt].dx(dr), (m*1j*v[dt]+v[dr])/r]])

	def div(self,v,m,i=0):
		r=self.r
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		return r*v[dx].dx(dx) + (1+i)*v[dr] + r*v[dr].dx(dr) + m*1j*v[dt]

	def div_nor(self,v,m):
		r=self.r
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		if len(v.ufl_shape)==1:
			return v[dx].dx(dx) + (r*v[dr]).dx(dr)/r + m*1j*v[dt]/r
		else:
			return ufl.as_vector([v[dx,dx].dx(dx)+v[dr,dx].dx(dr)+(v[dr,dx]+m*1j*v[dt,dx])/r,
								  v[dx,dr].dx(dx)+v[dr,dr].dx(dr)+(v[dr,dr]+m*1j*v[dt,dr]-v[dt,dt])/r,
								  v[dx,dt].dx(dx)+v[dr,dt].dx(dr)+(v[dr,dt]+m*1j*v[dt,dt]+v[dt,dr])/r])
	
	def r2vis(self,v,m):
		r=self.r
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		nu=1/self.Re+self.nut
		return ufl.as_vector([2*r**2*(nu*v[dx].dx(dx)).dx(dx)			  +r*(r*nu*(v[dr].dx(dx)+v[dx].dx(dr))).dx(dr)+m*1j*nu*(r*v[dt].dx(dx)+m*1j*v[dx]),
							  r**2*(nu*(v[dr].dx(dx)+v[dx].dx(dr))).dx(dx)+r*(2*r*nu*v[dr].dx(dr)).dx(dr)			  +m*1j*nu*(r*v[dt].dx(dr)+m*1j*v[dr])-2*m*1j*nu*v[dt],
							  r  *(nu*(r*v[dt].dx(dx)+m*1j*v[dx])).dx(dx) +r*(nu*(r*v[dt].dx(dr)+m*1j*v[dr])).dx(dr)-2*m**2*nu*v[dt]					  +nu*(r*v[dt].dx(dr)+m*1j*v[dr])])

	def r2vis_full(self,v,m):
		r=self.r
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		nu=1/self.Re+self.nut
		return ufl.as_vector([2*r**2*(nu.dx(dx)*v[dx].dx(dx)+nu*v[dx].dx(dx).dx(dx))			  +r*(nu*(v[dr].dx(dx)+v[dx].dx(dr))+r*nu.dx(dr)*(v[dr].dx(dx)+v[dx].dx(dr))+r*nu*(v[dr].dx(dx).dx(dr)+v[dx].dx(dr).dx(dr)))+m*1j*nu*(r*v[dt].dx(dx)+m*1j*v[dx]),
							  r**2*(nu.dx(dx)*(v[dr].dx(dx)+v[dx].dx(dr))+nu*(v[dr].dx(dx).dx(dx)+v[dx].dx(dr).dx(dx)))+r*(2*nu*v[dr].dx(dr)+2*r*nu.dx(dr)*v[dr].dx(dr)+2*r*nu*v[dr].dx(dr).dx(dr))			  +m*1j*nu*(r*v[dt].dx(dr)+m*1j*v[dr])-2*m*1j*nu*v[dt],
							  r  *(nu.dx(dx)*(r*v[dt].dx(dx)+m*1j*v[dx])+nu*(r*v[dt].dx(dx).dx(dx)+m*1j*v[dx].dx(dx))) +r*(nu.dx(dr)*(r*v[dt].dx(dr)+m*1j*v[dr])+nu*(v[dt].dx(dr)+r*v[dt].dx(dr).dx(dr)+m*1j*v[dr].dx(dr)))-2*m**2*nu*v[dt]					  +nu*(r*v[dt].dx(dr)+m*1j*v[dr])])

	# Helper
	def loadBaseflow(self,S,Re):
		typ=self.C*"complex/"+(1-self.C)*"real/"
		loadStuff(self.u_path+typ,['S','Re'],[S,Re],self.U.vector,self.io)
		loadStuff(self.p_path+typ,['S','Re'],[S,Re],self.P.vector,self.io)
		self.nutf(self,S,Re)

	def saveBaseflow(self,str):
		typ=self.C*"complex/"+(1-self.C)*"real/"
		dirCreator(self.u_path)
		dirCreator(self.p_path)
		saveStuff(self.u_path+typ,"u"+str+".dat",self.U.vector,self.io)
		saveStuff(self.p_path+typ,"p"+str+".dat",self.P.vector,self.io)

	def computeSUPG(self,m):
		# Split Arguments
		v,_ = ufl.split(self.test)
		U   = self.U # Baseflow

		h,nu=self.h,1/self.Re+self.nut

		# Weird Johann local tau (SUPG stabilisation)
		n=ufl.sqrt(ufl.inner(U,U))
		Pe=ufl.real(n*h/nu)
		cPe=ufl.conditional(ufl.le(Pe,3),Pe/3,1)

		self.SUPG = cPe*h/2/n*self.grd(v,m)*U # Streamline Upwind Petrov Galerkin
	
	# Heart of this entire code
	def navierStokes(self) -> ufl.Form:
		# Shortforms
		r,nu=self.r,1/self.Re+self.nut
		div,grd=self.div,self.grd
		
		# Functions
		U,P=self.U,self.P
		v,s=ufl.split(self.test)
		
		# Mass (variational formulation)
		F  = ufl.inner(	   div(U,0),    r**2*s)
		# Momentum (different test functions and IBP)
		F += ufl.inner(    grd(U,0)*U,  r**2*v) # Convection
		F -= ufl.inner(	  	 r*P,	   div(r*v,0,1)) # Pressure
		F += ufl.inner(nu*(grd(U,0)+
					   	   grd(U,0).T),grd(r*v,0,1)) # Diffusion (grad u.T significant with nut)
		return F*ufl.dx
		
	# Not automatic because of convection term
	def linearisedNavierStokes(self,m:int) -> ufl.Form:
		# Shortforms
		r,nu=self.r,1/self.Re+self.nut
		div,grd=self.div,self.grd
		SUPG,r2vis=self.SUPG,self.r2vis_full
		
		# Functions
		u, p=ufl.split(self.trial)
		U   = self.U # Baseflow
		v, s=ufl.split(self.test)
		"""
		# Mass (variational formulation)
		F  = ufl.inner(	   div(u,m),     r*s)
		# Momentum (different test functions and IBP)
		F += ufl.inner(	   grd(U,0)*u,   r*v) # Convection
		F += ufl.inner(	   grd(u,m)*U,   r*v)
		F -= ufl.inner(	       p,    r*div(v,m,1)) # Pressure
		F += ufl.inner(nu*(grd(u,m)+
					   	   grd(u,m).T),grd(v,m,1)) # Diffusion (grad u.T significant with nut)
		"""
		# Mass (variational formulation)
		F  = ufl.inner(	   div(u,m),     r*s)
		# Momentum (different test functions and IBP)
		F += ufl.inner(	   grd(U,0)*u,   r*v+r*SUPG) # Convection
		F += ufl.inner(	   grd(u,m)*U,   r*v+r*SUPG)
		F -= ufl.inner(	       p,    r*div(v,m,1)) # Pressure
		#F += ufl.inner(	   grd(p,m),	   	 r*SUPG)
		F += ufl.inner(nu*(grd(u,m)+
					   	   grd(u,m).T),grd(v,m,1)) # Diffusion (grad u.T significant with nut)
		#F -= ufl.inner(  r2vis(u,m),		   SUPG)
		"""
		# Mass (variational formulation)
		F  = ufl.inner(	   div(u,m),    s)
		# Momentum (different test functions and IBP)
		F += ufl.inner(	   grd(U,0)*u,	   v +SUPG) # Convection
		F += ufl.inner(	   grd(u,m)*U,	   v +SUPG)
		F -= ufl.inner(	       p,      div(v,m)) # Pressure
		F += ufl.inner(	   grd(p,m),   		  SUPG)
		F += ufl.inner(nu*(grd(u,m)+
					   	   grd(u,m).T),grd(v,m)) # Diffusion (grad u.T significant with nut)
		F += ufl.inner(div(nu*(grd(u,m)+
					   	   	   grd(u,m).T),m),SUPG)
		F*=r
		# Mass (variational formulation)
		F  = ufl.inner(	   div(u,m),    s)
		# Momentum (different test functions and IBP)
		F += ufl.inner(	   grd(U,0)*u,  v) # Convection
		F += ufl.inner(	   grd(u,m)*U,  v)
		F -= ufl.inner(	       p,      div(v,m)) # Pressure
		F += ufl.inner(nu*(grd(u,m)+
					   	   grd(u,m).T),grd(v,m)) # Diffusion (grad u.T significant with nut)
		F*=r
		"""
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

	def printStuff(self,dir:str,name:str,fun:Function) -> None:
		dirCreator(dir)
		with dfx.io.XDMFFile(comm, dir+name+".xdmf", "w") as xdmf:
			xdmf.write_mesh(self.mesh)
			xdmf.write_function(fun)

	# Converters
	def datToNpy(self,fi:str,fo:str,fun:Function) -> None:
		fun.vector = self.io.readBinaryFile(fi)[0]
		fun.x.scatter_forward()
		np.save(fo,fun.x.array)

	def datToNpyAll(self) -> None:
		for fun,path in [(self.U,self.u_path),(self.P,self.p_path),(self.nut,self.nut_path)]:
			if os.path.isdir(path+"real/"):
				file_names = [f for f in os.listdir(path) if f[-3:]=="dat"]
				dirCreator(path+"npy/")
				for file_name in file_names:
					self.datToNpy(path+"real/"+file_name,
								  path+"npy/"+file_name[:-4]+".npy",fun)
				if p0: shutil.rmtree(path+"real/")

	def npyToDat(self,fi:str,fo:str,fun:Function) -> None:
		fun.x.array.real=np.load(fi,allow_pickle=True)
		fun.x.scatter_forward()
		solnAsPetscBiIOVec = fun.vector.array_w.view(PetscBinaryIO.Vec)
		self.io.writeBinaryFile(fo, [solnAsPetscBiIOVec])
	
	def npyToDatAll(self) -> None:
		for fun,path in [(self.U,self.u_path),(self.P,self.p_path),(self.nut,self.nut_path)]:
			if os.path.isdir(path+"npy/"):
				file_names = [f for f in os.listdir(path+"npy/")]
				dirCreator(path+"complex/")
				for file_name in file_names:
					match = re.search(r'_p=([0-9]*)',file_name)
					if comm.rank==int(match.group(1)):
						self.npyToDat(path+"npy/"+file_name,
									  path+"complex/"+file_name[:-4]+".dat",fun)
				if p0: shutil.rmtree(path+"npy/")
	
	# Quick check functions
	def sanityCheckU(self):
		self.printStuff("./","sanity_check_u",self.U)

	def sanityCheck(self):
		self.printStuff("./","sanity_check_u",  self.U)
		self.printStuff("./","sanity_check_p",  self.P)
		self.printStuff("./","sanity_check_nut",self.nut)

	def sanityCheckBCs(self):
		v=Function(self.TH)
		v.vector.zeroEntries()
		v.x.array[self.dofs]=np.ones(self.dofs.size)
		u,_=v.split()
		self.printStuff("./","sanity_check_bcs",u)