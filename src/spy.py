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

def dirCreator(path:str) -> None:
	if not os.path.isdir(path):
		if p0: os.mkdir(path)
	comm.barrier() # Wait for all other processors

def checkComm(f:str) -> bool:
	match = re.search(r'n=(\d*)',f)
	if int(match.group(1))!=comm.size: return False
	match = re.search(r'p=([0-9]*)',f)
	if int(match.group(1))!=comm.rank: return False
	return True

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

# Naive save with dir creation
def saveStuff(dir:str,name:str,fun:Function) -> None:
	dirCreator(dir)
	proc_name=dir+name.replace('.',',')+f"_n={comm.size:d}_p={comm.rank:d}"
	fun.x.scatter_forward()
	np.save(proc_name,fun.x.array)
	if p0: print("Saved "+proc_name+".npy",flush=True)

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
	if p0: print("Loading "+closest_file_name,flush=True)
	fun.x.array[:]=np.load(closest_file_name,allow_pickle=True)
	fun.x.scatter_forward()

# Swirling Parallel Yaj
class SPY:
	def __init__(self, params:dict, datapath:str, mesh_name:str, direction_map:dict) -> None:
		# Direction dependant
		self.direction_map=direction_map
		# Solver parameters (Newton mostly, but also eig)
		self.params=params

		# Paths
		self.case_path	   ='/home/shared/cases/'+datapath+'/'
		self.baseflow_path =self.case_path+'baseflow/'
		self.q_path	 	   =self.baseflow_path+'q/'
		self.nut_path	   =self.baseflow_path+'nut/'
		self.print_path	   =self.baseflow_path+'print/'
		self.resolvent_path=self.case_path+'resolvent/'
		self.eig_path	   =self.case_path+'eigenvalues/'

		# Mesh from file
		meshpath=self.case_path+"mesh/"+mesh_name+".xdmf"
		with dfx.io.XDMFFile(comm, meshpath, "r") as file:
			self.mesh = file.read_mesh(name="Grid")
		if p0: print("Loaded "+meshpath,flush=True)
		self.defineFunctionSpaces()

		# BCs essentials
		self.dofs = np.empty(0,dtype=np.int32)
		self.bcs  = []

	# TO be rerun if mesh changes
	def defineFunctionSpaces(self) -> None:
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
		self.u, self.p = ufl.TrialFunctions(self.TH)
		self.v, self.s = ufl.TestFunctions( self.TH)
		# Initialisation of baseflow
		self.Q = Function(self.TH)
		# Collapsed subspaces
		self.U, self.P, self.Nu = Function(self.TH0), Function(self.TH1), Function(self.TH1)

	# Helper
	def loadBaseflow(self,Re:int,S:float,loadNu=True) -> None:
		loadStuff(self.q_path,  {'Re':Re,'S':S},self.Q)
		if loadNu: loadStuff(self.nut_path,{'Re':Re,'S':S},self.Nu)
		
	def saveBaseflow(self,Re:int,S:float): saveStuff(self.q_path,f"q_Re={Re:d}_S={S:.1f}".replace('.',','),self.Q)
	
	# Heart of this entire code
	def navierStokes(self) -> ufl.Form:
		# Shortforms
		r, v, s = self.r, self.v, self.s
		# Functions
		U, P = ufl.split(self.Q)
		nu = 1/self.Re+self.Nu
		# More shortforms
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		dv,gd=lambda v: div(r,dx,dr,dt,v,0),lambda v: grd(r,dx,dr,dt,v,0)
		# Mass (variational formulation)
		F  = ufl.inner(dv(U),   s)
		# Momentum (different test functions and IBP)
		F += ufl.inner(gd(U)*U, v) # Convection
		F -= ufl.inner(	  P, dv(v)) # Pressure
		F += ufl.inner(gd(U)+gd(U).T,
							 gd(v))*nu # Diffusion (grad u.T significant with nut)
		return F*r*ufl.dx
	
	# Not automatic because of convection term
	def linearisedNavierStokes(self,m:int) -> ufl.Form:
		# Shortforms
		r, u, p, v, s = self.r, self.u, self.p, self.v, self.s
		# Functions
		U, _ = ufl.split(self.Q) # Baseflow
		nu = 1/self.Re + self.Nu
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
							   gd(v,m))*nu # Diffusion (grad u.T significant with nut)
		return F*r*ufl.dx
	
	# Evaluate velocity at provided points
	def eval(self,f,proj_pts,ref_pts=None) -> np.array:
		proj_pts = np.hstack((proj_pts,np.zeros((proj_pts.shape[0],1))))
		if ref_pts is None:
			ref_pts=proj_pts
			return_pts=False
		else:
			return_pts=True
		bbtree = dfx.geometry.BoundingBoxTree(self.mesh, 2)
		local_proj, local_ref, local_cells = [], [], []
		# Find cells whose bounding-box collide with the the points
		cell_candidates = dfx.geometry.compute_collisions(bbtree, proj_pts)
		# Choose one of the cells that contains the point
		colliding_cells = dfx.geometry.compute_colliding_cells(self.mesh, cell_candidates, proj_pts)
		for i, pt in enumerate(proj_pts):
			if len(colliding_cells.links(i))>0:
				local_proj.append(pt)
				local_ref.append(ref_pts[i])
				local_cells.append(colliding_cells.links(i)[0])
		# Actual evaluation
		if len(local_proj)!=0: V = f.eval(local_proj, local_cells)
		else: V = None
		# Gather data and points
		V = comm.gather(V, root=0)
		ref_pts = comm.gather(local_ref, root=0)
		if p0:
			V = np.hstack([v.flatten() for v in V if v is not None])
			ref_pts = np.vstack([np.array(pts) for pts in ref_pts if len(pts)>0])
			# Filter ghost values
			ref_pts, ids_u = np.unique(ref_pts, return_index=True, axis=0)
			# Return relevant evaluation points
			if return_pts: return ref_pts, V[ids_u]
			return V[ids_u]
		if return_pts: return None, None

	# Code factorisation
	def constantBC(self, direction:chr, boundary:bool, value:float=0) -> tuple:
		subspace=self.TH.sub(0).sub(self.direction_map[direction])
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

	def applyHomogeneousBCs(self, tup:list) -> None:
		for marker,directions in tup:
			for direction in directions:
				dofs,bcs=self.constantBC(direction,marker)
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

		dx,dr=self.direction_map['x'],self.direction_map['r']
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

	def sanityCheckBCs(self,str=""):
		v=Function(self.TH)
		v.vector.zeroEntries()
		v.x.array[self.dofs]=np.ones(self.dofs.size)
		v,_=v.split()
		self.printStuff("./","sanity_check_bcs"+str,v)