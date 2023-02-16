# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import os, ufl, re
import numpy as np
import dolfinx as dfx
from dolfinx.fem import Function
from mpi4py.MPI import COMM_WORLD as comm

p0=comm.rank==0

# Cylindrical operators
def grd(r,dx:int,dr:int,dt:int,v,m:int):
	if len(v.ufl_shape)==0: return ufl.as_vector([v.dx(dx), v.dx(dr), m*1j*v/r])
	return ufl.as_tensor([[v[dx].dx(dx), v[dx].dx(dr),  m*1j*v[dx]		 /r],
						  [v[dr].dx(dx), v[dr].dx(dr), (m*1j*v[dr]-v[dt])/r],
						  [v[dt].dx(dx), v[dt].dx(dr), (m*1j*v[dt]+v[dr])/r]])

def div(r,dx:int,dr:int,dt:int,v,m:int):
	if len(v.ufl_shape)==1: return v[dx].dx(dx) + (r*v[dr]).dx(dr)/r + m*1j*v[dt]/r
	return ufl.as_vector([v[dx,dx].dx(dx)+v[dr,dx].dx(dr)+v[dr,dx]+m*1j*v[dt,dx]/r,
						  v[dx,dr].dx(dx)+v[dr,dr].dx(dr)+v[dr,dr]+(m*1j*v[dt,dr]-v[dt,dt])/r,
						  v[dx,dt].dx(dx)+v[dr,dt].dx(dr)+v[dr,dt]+(m*1j*v[dt,dt]+v[dt,dr])/r])

def crl(r,dx:int,dr:int,dt:int,mesh:ufl.Mesh,v,m:int,i:int=0):
	return ufl.as_vector([(i+1)*v[dt]		+r*v[dt].dx(dr)-m*dfx.fem.Constant(mesh, 1j)*v[dr],
    m*dfx.fem.Constant(mesh,1j)*v[dx]		-  v[dt].dx(dx),
								v[dr].dx(dx)-i*v[dx]-v[dx].dx(dr)])

def checkComm(f:str):
	match = re.search(r'n=(\d*)',f)
	if int(match.group(1))!=comm.size: return False
	match = re.search(r'p=([0-9]*)',f)
	if int(match.group(1))!=comm.rank: return False
	return True

def dirCreator(path:str):
	if not os.path.isdir(path):
		if p0: os.mkdir(path)
	comm.barrier() # Wait for all other processors

# Simple handler
def meshConvert(path:str,cell_type:str='triangle',prune=True) -> None:
	import meshio #pip3 install h5py meshio
	gmsh_mesh = meshio.read(path+".msh")
	# Write it out again
	ps = gmsh_mesh.points[:,:(3-prune)]
	cs = gmsh_mesh.get_cells_type(cell_type)
	dolfinx_mesh = meshio.Mesh(points=ps, cells={cell_type: cs})
	meshio.write(path+".xdmf", dolfinx_mesh)
	print("Mesh "+path+".msh converted to "+path+".xdmf !",flush=True)

# Memoisation routine - find closest in param
def findStuff(path:str,params:dict,format=lambda f:True,distributed=True):
	closest_file_name=path
	file_names = [f for f in os.listdir(path) if format(f)]
	d=np.infty
	for file_name in file_names:
		if not distributed or checkComm(file_name): # Lazy evaluation !
			fd=0 # Compute distance according to all params
			for param in params:
				match = re.search(param+r'=(\d*(,|e|-|j|\+)?\d*)',file_name)
				param_file = float(match.group(1).replace(',','.')) # Take advantage of file format
				fd += abs(params[param]-param_file)
			if fd<d: d,closest_file_name=fd,path+file_name
	return closest_file_name

def loadStuff(path:str,params:dict,fun:Function) -> None:
	closest_file_name=findStuff(path,params,lambda f: f[-3:]=="npy")
	fun.x.array[:]=np.load(closest_file_name,allow_pickle=True)
	fun.x.scatter_forward()
	# Loading eddy viscosity too
	if p0: print("Loaded "+closest_file_name,flush=True)

# Naive save with dir creation
def saveStuff(dir:str,name:str,fun:Function) -> None:
	dirCreator(dir)
	proc_name=dir+name.replace('.',',')+f"_n={comm.size:d}_p={comm.rank:d}"
	fun.x.scatter_forward()
	np.save(proc_name,fun.x.array)
	if p0: print("Saved "+proc_name+".npy",flush=True)

# Swirling Parallel Yaj
class SPY:
	def __init__(self, params:dict, datapath:str, mesh_name:str, direction_map:dict) -> None:
		# Direction dependant
		self.direction_map=direction_map
		# Solver parameters (Newton mostly, but also eig)
		self.params=params

		# Paths
		self.case_path	   ='/home/shared/cases/'+datapath
		self.baseflow_path =self.case_path+'baseflow/'
		self.q_path	 	   =self.baseflow_path+'q/'
		self.nut_path	   =self.baseflow_path+'nut/'
		self.print_path	   =self.baseflow_path+'print/'
		self.resolvent_path=self.case_path+'resolvent/'
		self.eig_path	   =self.case_path+'eigenvalues/'

		# Mesh from file
		meshpath=self.case_path+mesh_name+".xdmf"
		with dfx.io.XDMFFile(comm, meshpath, "r") as file:
			self.mesh = file.read_mesh(name="Grid")
		if p0: print("Loaded "+meshpath,flush=True)
		self.defineFunctionSpaces()

		# BCs essentials
		self.dofs = np.empty(0,dtype=np.int32)
		self.bcs  = []

	# TO be rerun if mesh changes
	def defineFunctionSpaces(self):
		# Extraction of r
		self.r = ufl.SpatialCoordinate(self.mesh)[self.direction_map['r']]
		# Finite elements & function spaces
		FE_vector =ufl.VectorElement("CG",self.mesh.ufl_cell(),2,3)
		FE_scalar =ufl.FiniteElement("CG",self.mesh.ufl_cell(),1)
		FE_scalar2=ufl.FiniteElement("CG",self.mesh.ufl_cell(),2)
		#Constant =ufl.FiniteElement("Real",self.mesh.ufl_cell(),0)
		self.TH0 = dfx.fem.FunctionSpace(self.mesh,FE_vector)
		self.TH1 = dfx.fem.FunctionSpace(self.mesh,FE_scalar)
		self.TH2 = dfx.fem.FunctionSpace(self.mesh,FE_scalar2)
		# Taylor Hodd elements ; stable element pair + eddy viscosity
		self.TH = dfx.fem.FunctionSpace(self.mesh,FE_vector*FE_scalar)
		self.TH0c, self.TH_to_TH0 = self.TH.sub(0).collapse()
		self.TH1c, self.TH_to_TH1 = self.TH.sub(1).collapse()
		# Test & trial functions
		self.trial = ufl.TrialFunction(self.TH)
		self.test  = ufl.TestFunction( self.TH)
		# Initialisation of baseflow
		self.Q = Function(self.TH)
		# Collapsed subspaces
		self.U, self.P, self.Nu = Function(self.TH0), Function(self.TH1), Function(self.TH2)

	# Helper
	def loadBaseflow(self,Re:int,S:float):
		loadStuff(self.q_path,  {'Re':Re,'S':S},self.Q)
		loadStuff(self.nut_path,{'Re':Re,'S':S},self.Nu)

	def saveBaseflow(self,Re:int,S:float): saveStuff(self.q_path,f"q_S={S:.1f}_Re={Re:d}".replace('.',','),self.Q)
	
	# Heart of this entire code
	def navierStokes(self) -> ufl.Form:
		# Shortforms
		r, Re = self.r, self.Re
		# Functions
		U, P = ufl.split(self.Q)
		Nu = self.Nu
		v, s  = ufl.split(self.test)
		# More shortforms
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		dv,gd=lambda v: div(r,dx,dr,dt,v,0),lambda v: grd(r,dx,dr,dt,v,0)
		# Mass (variational formulation)
		F  = ufl.inner(dv(U),   s)
		# Momentum (different test functions and IBP)
		F += ufl.inner(gd(U)*U, v) # Convection
		F -= ufl.inner(	  P, dv(v)) # Pressure
		F += ufl.inner(gd(U)+gd(U).T,
							 gd(v))*(1/Re+Nu) # Diffusion (grad u.T significant with nut)
		return F*r*ufl.dx
	
	# Not automatic because of convection term
	def linearisedNavierStokes(self,m:int) -> ufl.Form:
		# Shortforms
		r,Re=self.r,self.Re
		# Functions
		u, p = ufl.split(self.trial)
		U, _ = ufl.split(self.Q) # Baseflow
		Nu = self.Nu 
		v, s = ufl.split(self.test)
		# More shortforms
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		dv,gd=lambda v,m: div(r,dx,dr,dt,v,m),lambda v,m: grd(r,dx,dr,dt,v,m)
		# Mass (variational formulation)
		F  = ufl.inner(dv(u,m),   s)
		# Momentum (different test functions and IBP)
		F += ufl.inner(gd(U,0)*u, v) # Convection
		F += ufl.inner(gd(u,m)*U, v)
		F -= ufl.inner(   p,   dv(v,m)) # Pressure
		F += ufl.inner(gd(u,m)+gd(u,m).T,
							   gd(v,m))*(1/Re+Nu) # Diffusion (grad u.T significant with nut)
		return F*r*ufl.dx

	# Code factorisation
	def constantBC(self, direction:chr, boundary:bool, value:float=0, subspace_i:int=0) -> tuple:
		subspace=self.TH.sub(subspace_i)
		if subspace_i==0: subspace=subspace.sub(self.direction_map[direction])
		subspace_collapsed,_=subspace.collapse()
		# Compute unflattened DoFs (don't care for flattened ones)
		dofs = dfx.fem.locate_dofs_geometrical((subspace, subspace_collapsed), boundary)
		cst = Function(subspace_collapsed)
		cst.interpolate(lambda x: np.ones_like(x[0])*value)
		# Actual BCs
		bcs = dfx.fem.dirichletbc(cst, dofs, subspace) # u_i=value at boundary
		return dofs[0],bcs

	# Encapsulation	
	def applyBCs(self, dofs:np.ndarray, bcs) -> None:
		self.dofs=np.union1d(dofs,self.dofs)
		self.bcs.append(bcs)

	def applyHomogeneousBCs(self, tup:list, subspace_i:int=0) -> None:
		for marker,directions in tup:
			for direction in directions:
				dofs,bcs=self.constantBC(direction,marker,subspace_i=subspace_i)
				self.applyBCs(dofs,bcs)

	def printStuff(self,dir:str,name:str,fun:Function) -> None:
		dirCreator(dir)
		with dfx.io.XDMFFile(comm, dir+name.replace('.',',')+".xdmf", "w") as xdmf:
			xdmf.write_mesh(self.mesh)
			xdmf.write_function(fun)
		if p0: print("Printed "+dir+name.replace('.',',')+".xdmf",flush=True)
	
	# Quick check functions
	def sanityCheckU(self,app=""):
		U,_=self.Q.split()
		self.printStuff("./","sanity_check_u"+app,U)

	def sanityCheck(self,app=""):
		self.U.x.array[:]=self.Q.x.array[self.TH_to_TH0]
		self.P.x.array[:]=self.Q.x.array[self.TH_to_TH1]
		#self.Nu.x.array[:]=self.Nu.x.array[:]

		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		expr=dfx.fem.Expression(self.U[dx].dx(dx) + (self.r*self.U[dr]).dx(dr)/self.r,
								self.TH1.element.interpolation_points())
		div = Function(self.TH1)
		div.interpolate(expr)
		self.printStuff("./","sanity_check_div"+app,div)

		FE = ufl.FiniteElement("DG",self.mesh.ufl_cell(),0)
		W = dfx.fem.FunctionSpace(self.mesh,FE)
		p = Function(W)
		p.x.array[:]=comm.rank
		self.printStuff("./","sanity_check_partition"+app,p)
		
		self.printStuff("./","sanity_check_u"+app,  self.U)
		self.printStuff("./","sanity_check_p"+app,  self.P)
		# nut may not be a Function
		try: self.printStuff("./","sanity_check_nut"+app,self.Nu)
		except TypeError: pass

	def sanityCheckBCs(self):
		v=Function(self.TH)
		v.vector.zeroEntries()
		v.x.array[self.dofs]=np.ones(self.dofs.size)
		v,_=v.split()
		self.printStuff("./","sanity_check_bcs",v)