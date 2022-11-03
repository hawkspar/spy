# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import os, ufl, re
import numpy as np
import dolfinx as dfx
from petsc4py import PETSc as pet
from mpi4py.MPI import COMM_WORLD as comm
from dolfinx.fem import FunctionSpace, Function

p0=comm.rank==0

def checkComm(f:str):
	try:
		match = re.search(r'_n=(\d*)',f)
		if int(match.group(1))!=comm.size: return False
		match = re.search(r'_p=([0-9]*)',f)
		if int(match.group(1))!=comm.rank: return False
	except AttributeError: pass
	return True

def dirCreator(path:str):
	if not os.path.isdir(path):
		if p0: os.mkdir(path)
	comm.barrier() # Wait for all other processors

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
def findStuff(path:str,keys:list,params:list,format):
	closest_file_name=path
	file_names = [f for f in os.listdir(path) if format(f)]
	d=np.infty
	for file_name in file_names:
		if checkComm(file_name):
			fd=0 # Compute distance according to all params
			for param,key in zip(params,keys):
				match = re.search(r'_'+key+r'=(\d*(\.|,|e|-)?\d*)',file_name)
				param_file = float(match.group(1).replace(',','.')) # Take advantage of file format
				fd += abs(param-param_file)
			if fd<d: d,closest_file_name=fd,path+file_name
	return closest_file_name

def loadStuff(path:str,keys:list,params:list,fun:Function) -> None:
	closest_file_name=findStuff(path,keys,params,lambda f: f[-3:]=="npy")
	fun.x.array[:]=np.load(closest_file_name,allow_pickle=True)
	fun.x.scatter_forward()
	# Loading eddy viscosity too
	if p0: print("Loaded "+closest_file_name,flush=True)

# Naive save with dir creation
def saveStuff(dir:str,name:str,fun:Function) -> None:
	dirCreator(dir)
	proc_name=dir+name+f"_n={comm.size:d}_p={comm.rank:d}"
	fun.x.scatter_forward()
	np.save(proc_name,fun.x.array)
	if p0: print("Saved "+proc_name,flush=True)

# Swirling Parallel Yaj
class SPY:
	def __init__(self, params:dict, datapath:str, direction_map:dict, forcingIndicator=None) -> None:
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

		# Extraction of r
		self.r = ufl.SpatialCoordinate(self.mesh)[direction_map['r']]
		# Local cell number
		tdim = self.mesh.topology.dim
		num_cells = self.mesh.topology.index_map(tdim).size_local + self.mesh.topology.index_map(tdim).num_ghosts
	
		# Finite elements & function spaces
		FE_vector  =ufl.VectorElement("CG",self.mesh.ufl_cell(),2,3)
		FE_scalar  =ufl.FiniteElement("CG",self.mesh.ufl_cell(),1)
		FE_constant=ufl.FiniteElement("DG",self.mesh.ufl_cell(),0)
		self.TH0 = FunctionSpace(self.mesh,FE_vector)
		self.TH1 = FunctionSpace(self.mesh,FE_scalar)
		self.TH  = FunctionSpace(self.mesh,ufl.MixedElement(FE_vector,FE_scalar)) # Taylor Hodd elements ; stable element pair
		W = FunctionSpace(self.mesh,FE_constant)
		self.TH0c, self.TH0_to_TH = self.TH.sub(0).collapse()
		self.TH1c, self.TH1_to_TH = self.TH.sub(1).collapse()
		
		# Test & trial functions
		self.trial = ufl.TrialFunction(self.TH)
		self.test  = ufl.TestFunction( self.TH)
		
		# Initialisation of baseflow
		self.Q = Function(self.TH)
		# Collapsed subspaces
		self.U,self.P = Function(self.TH0), Function(self.TH1)
		# Local mesh size
		self.h = Function(W)
		self.h.x.array[:]=dfx.cpp.mesh.h(self.mesh, tdim, range(num_cells))
	
		# Forcing localisation
		if forcingIndicator==None: self.indic=1
		else:
			self.indic = Function(W)
			self.indic.interpolate(forcingIndicator)

		# BCs essentials
		self.dofs = np.empty(0,dtype=np.int32)
		self.bcs  = []

	# Jet geometry
	def symmetry(self, x:ufl.SpatialCoordinate) -> np.ndarray:
		return np.isclose(x[self.direction_map['r']],0,self.params['atol']) # Axis of symmetry at r=0
	
	# Vanilla operators
	def grd_nor(self,v,m):
		r=self.r
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		if len(v.ufl_shape)==0:
			return ufl.as_vector([v.dx(dx), v.dx(dr), m*1j*v/r])
		return ufl.as_tensor([[v[dx].dx(dx), v[dx].dx(dr),  m*1j*v[dx]		 /r],
							  [v[dr].dx(dx), v[dr].dx(dr), (m*1j*v[dr]-v[dt])/r],
							  [v[dt].dx(dx), v[dt].dx(dr), (m*1j*v[dt]+v[dr])/r]])

	def div_nor(self,v,m):
		r=self.r
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		if len(v.ufl_shape)==1:
			return v[dx].dx(dx) + (r*v[dr]).dx(dr)/r + m*1j*v[dt]/r
		else:
			return ufl.as_vector([v[dx,dx].dx(dx)+v[dr,dx].dx(dr)+(v[dr,dx]+m*1j*v[dt,dx])/r,
								  v[dx,dr].dx(dx)+v[dr,dr].dx(dr)+(v[dr,dr]+m*1j*v[dt,dr]-v[dt,dt])/r,
								  v[dx,dt].dx(dx)+v[dr,dt].dx(dr)+(v[dr,dt]+m*1j*v[dt,dt]+v[dt,dr])/r])

	# Gradient with r multiplication
	def grd(self,v,m,i=0):
		r=self.r
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		if len(v.ufl_shape)==0: return ufl.as_vector([r*v.dx(dx), i*v+r*v.dx(dr), m*1j*v])
		return ufl.as_tensor([[r*v[dx].dx(dx), i*v[dx]+r*v[dx].dx(dr), m*1j*v[dx]],
							  [r*v[dr].dx(dx), i*v[dr]+r*v[dr].dx(dr), m*1j*v[dr]-v[dt]],
							  [r*v[dt].dx(dx), i*v[dt]+r*v[dt].dx(dr), m*1j*v[dt]+v[dr]]])

	def div(self,v,m,i=0):
		r=self.r
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		return r*v[dx].dx(dx) + (1+i)*v[dr] + r*v[dr].dx(dr) + m*1j*v[dt]

	def r2vis(self,v,m):
		r=self.r
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		nu=1/self.Re+self.nut
		return ufl.as_vector([2*r**2*(nu*v[dx].dx(dx)).dx(dx)			  +r*(r*nu*(v[dr].dx(dx)+v[dx].dx(dr))).dx(dr)+m*1j*nu*(r*v[dt].dx(dx)+m*1j*v[dx]),
							  r**2*(nu*(v[dr].dx(dx)+v[dx].dx(dr))).dx(dx)+r*(2*r*nu*v[dr].dx(dr)).dx(dr)			  +m*1j*nu*(r*v[dt].dx(dr)+m*1j*v[dr])-2*m*1j*nu*v[dt],
							  r  *(nu*(r*v[dt].dx(dx)+m*1j*v[dx])).dx(dx) +r*(nu*(r*v[dt].dx(dr)+m*1j*v[dr])).dx(dr)-2*m**2*nu*v[dt]					  +nu*(r*v[dt].dx(dr)+m*1j*v[dr])])

	def r2vis2(self,v,m):
		r=self.r
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		nu=1/self.Re+self.nut
		return ufl.as_vector([r**2*(nu*v[dx].dx(dx)).dx(dx)+r*(r*nu*v[dr].dx(dx)).dx(dr)+m*1j*nu*r*v[dt].dx(dx),
							  r**2*(nu*v[dx].dx(dr)).dx(dx)+r*(r*nu*v[dr].dx(dr)).dx(dr)+m*1j*nu*r*v[dt].dx(dr)-m*1j*nu*v[dt],
							  r   *m*1j*(nu*v[dx]).dx(dx)  +r*m*1j*(nu*v[dr]).dx(dr)	-m**2*nu*v[dt]		   +m*1j*nu*v[dr]])

	# Helper
	def loadBaseflow(self,Re,nut,S,p=False):
		# Load separately
		loadStuff(self.u_path,['S','nut','Re'],[S,nut,Re],self.U)
		if p: loadStuff(self.p_path,['S','nut','Re'],[S,nut,Re],self.P)
		# Write inside MixedElement
		self.Q.x.array[self.TH0_to_TH]=self.U.x.array
		self.Q.x.scatter_forward()
		if p:
			self.Q.x.array[self.TH1_to_TH]=self.P.x.array
			self.Q.x.scatter_forward()

	def saveBaseflow(self,str):
		self.Q.x.scatter_forward()
		# Write inside MixedElement
		self.U.x.array[:]=self.Q.x.array[self.TH0_to_TH]
		self.P.x.array[:]=self.Q.x.array[self.TH1_to_TH]
		dirCreator(self.u_path)
		dirCreator(self.p_path)
		saveStuff(self.u_path,"u"+str,self.U)
		saveStuff(self.p_path,"p"+str,self.P)
	
	# Pseudo-heat equation
	def smoothen(self, e:float, fun:Function, space:FunctionSpace, bcs, weak_bcs):
		u,v=ufl.TrialFunction(space),ufl.TestFunction(space)
		r=self.r
		grd=lambda v,i=0: self.grd(v,0,i)
		a=ufl.inner(u,v)*r**2
		a+=e*ufl.inner(grd(u),grd(v,1))
		L=ufl.inner(fun,v)
		pb = dfx.fem.petsc.LinearProblem(a*ufl.dx+weak_bcs(self,u,v), L*r**2*ufl.dx, bcs=bcs, petsc_options={"ksp_type": "cg", "pc_type": "gamg", "pc_factor_mat_solver_type": "mumps"})
		if p0: print("Smoothing started...",flush=True)
		res=pb.solve()
		res.x.scatter_forward()
		return res.x.array

	def stabilise(self,m):
		# Important ! Otherwise n is nonsense
		self.U.x.array[:]=self.Q.x.array[self.TH0_to_TH]
		# Split arguments
		U,_ = ufl.split(self.Q)
		v,_ = ufl.split(self.test)

		# Shorthands
		h,nu=self.h,1/self.Re+self.nut
		grd,div=self.grd,self.div

		# Weird Johann local tau (SUPG stabilisation)
		i=ufl.Index()
		n=ufl.sqrt(self.U[i]*self.U[i])
		Pe=ufl.real(n*h/2/nu)
		z=ufl.conditional(ufl.le(Pe,3),Pe/3,1)
		tau=z*h/2/n

		"""n=.5*ufl.sqrt(ufl.inner(U,U))
		Pe=ufl.real(n*h*self.Re)
		cPe=ufl.conditional(ufl.le(Pe,3),Pe/3,1)
		t=cPe*h/2/n"""
		#tau=1/ufl.sqrt((2*n/h)**2+(4*nu/h**2)**2)
		self.SUPG = tau*grd(v,m)*U # Streamline Upwind Petrov Galerkin
		"""gamma=(h/2)**2/4/3/t
		self.grd_div=gamma*div(v,m,1)"""
	
	# Heart of this entire code
	def navierStokes(self,weak_bcs,stabilise=False) -> ufl.Form:
		# Shortforms
		r, nu = self.r, 1/self.Re+self.nut
		div,grd=lambda v,i=0: self.div(v,0,i), lambda v,i=0: self.grd(v,0,i)
		r2vis, SUPG = lambda v: self.r2vis2(v,0), stabilise*self.SUPG
		
		# Functions
		U, P = ufl.split(self.Q)
		v, s = ufl.split(self.test)
		
		# Mass (variational formulation)
		F  = ufl.inner(div(U),   r*s)
		# Momentum (different test functions and IBP)
		F += ufl.inner(grd(U)*U,r*(v+SUPG)) # Convection
		F -= ufl.inner(	 r*P,  div(v,1)) # Pressure
		#F += ufl.inner(nu*(grd(U)+grd(U).T),grd(v,1)) # Diffusion (grad u.T significant with nut)
		F += ufl.inner(nu*grd(U),grd(v,1))
		if stabilise:
			F += ufl.inner(  grd(P),r*SUPG)
			F -= ufl.inner(r2vis(U),  SUPG)
		return F*ufl.dx+weak_bcs(self,U,P)
		
	# Not automatic because of convection term
	def linearisedNavierStokes(self,weak_bcs,m:int,stabilise=False) -> ufl.Form:
		# Shortforms
		r,nu=self.r,1/self.Re+self.nut
		div,grd=self.div,self.grd
		r2vis,SUPG=self.r2vis2,stabilise*self.SUPG
		
		# Functions
		u, p = ufl.split(self.trial)
		U, _ = ufl.split(self.Q) # Baseflow
		v, s = ufl.split(self.test)
		# Mass (variational formulation)
		F  = ufl.inner(div(u,m),  r*s)
		# Momentum (different test functions and IBP)
		F += ufl.inner(grd(U,0)*u,r*(v+SUPG)) # Convection
		F += ufl.inner(grd(u,m)*U,r*(v+SUPG))
		F -= ufl.inner(  r*p,    div(v,m,1)) # Pressure
		#F += ufl.inner(nu*(grd(u,m)+grd(u,m).T),grd(v,m,1)) # Diffusion (grad u.T significant with nut)
		F += ufl.inner(nu*grd(u,m),grd(v,m,1))
		if stabilise:
			F += ufl.inner(  grd(p,m),r*SUPG)
			F -= ufl.inner(r2vis(u,m),  SUPG)
		#F += ufl.inner(div(u,m),self.grd_div)
		return F*ufl.dx+weak_bcs(self,u,p,m)

	# Code factorisation
	def constantBC(self, direction:chr, boundary, value=0) -> tuple:
		sub_space=self.TH.sub(0).sub(self.direction_map[direction])
		sub_space_collapsed,_=sub_space.collapse()
		# Compute unflattened DoFs (don't care for flattened ones)
		dofs = dfx.fem.locate_dofs_geometrical((sub_space, sub_space_collapsed), boundary)
		cst = Function(sub_space_collapsed)
		cst.interpolate(lambda x: np.ones_like(x[0])*value)
		# Actual BCs
		bcs = dfx.fem.dirichletbc(cst, dofs, sub_space) # u_i=value at boundary
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
		self.U.x.array[:]=self.Q.x.array[self.TH0_to_TH]
		self.printStuff("./","sanity_check_u"+app,self.U)

	def sanityCheck(self,app=""):
		self.U.x.array[:]=self.Q.x.array[self.TH0_to_TH]
		self.P.x.array[:]=self.Q.x.array[self.TH1_to_TH]

		expr=dfx.fem.Expression(self.div_nor(self.U,0),self.TH1.element.interpolation_points())
		div = Function(self.TH1)
		div.interpolate(expr)
		self.printStuff("./","sanity_check_div"+app,div)

		FE = ufl.FiniteElement("DG",self.mesh.ufl_cell(),0)
		W = FunctionSpace(self.mesh,FE)
		p = Function(W)
		p.x.array[:]=comm.rank
		self.printStuff("./","sanity_check_partition"+app,p)
		
		self.printStuff("./","sanity_check_u"+app,  self.U)
		self.printStuff("./","sanity_check_p"+app,  self.P)
		# nut may not be a Function
		try: self.printStuff("./","sanity_check_nut"+app,self.nut)
		except TypeError: pass

	def sanityCheckBCs(self):
		v=Function(self.TH)
		v.vector.zeroEntries()
		v.x.array[self.dofs]=np.ones(self.dofs.size)
		v,_=v.split()
		self.printStuff("./","sanity_check_bcs",v)