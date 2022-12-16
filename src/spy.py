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

# SA constants
cb1,cb2 = .1355,.622
sig = 2./3.
kap = .41
cw1 = cb1/kap^2 + (1. + cb2)/sig
cw2,cw3 = .3,2
cv1 = 7.1
ct3,ct4 = 1.2,.5

def ft2(nu): return ct3*ufl.exp(-ct4*nu**2)
def ft2p(nu,dnu): return -2.0*ct4*ct3*nu*dnu*ufl.exp(-ct4*nu**2)

def fv1(c): return c**3/(c**3 + cv1**3)
def fv1p(c,dc): return 3.*cv1**3*c**2*dc/(c**3 + cv1**3)**2

def fv2(c): return 1. - c/(1. + c*fv1(c))
def fv2p(c,dc): return (c**2*fv1p(c,dc) - dc)/(1. + c*fv1(c))**2

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
def meshConvert(path:str,out:str,cell_type:str,prune=True) -> None:
	import meshio #pip3 install --no-binary=h5py h5py meshio
	gmsh_mesh = meshio.read(path+".msh")
	# Write it out again
	ps = gmsh_mesh.points[:,:(3-prune)]
	cs = gmsh_mesh.get_cells_type(cell_type)
	dolfinx_mesh = meshio.Mesh(points=ps, cells={cell_type: cs})
	meshio.write(out+".xdmf", dolfinx_mesh)
	print("Mesh "+path+".msh converted to "+out+".xdmf !",flush=True)

# Memoisation routine - find closest in param
def findStuff(path:str,keys:list,params:list,format=lambda f:True,distributed=True):
	closest_file_name=path
	file_names = [f for f in os.listdir(path) if format(f)]
	d=np.infty
	for file_name in file_names:
		if not distributed or checkComm(file_name): # Lazy evaluation !
			fd=0 # Compute distance according to all params
			for param,key in zip(params,keys):
				match = re.search(key+r'=(\d*(,|e|-|j|\+)?\d*)',file_name)
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
		FE_scalar2 =ufl.FiniteElement("CG",self.mesh.ufl_cell(),2)
		FE_constant=ufl.FiniteElement("DG",self.mesh.ufl_cell(),0)
		self.FS0 = FunctionSpace(self.mesh,FE_vector)
		self.FS1 = FunctionSpace(self.mesh,FE_scalar)
		self.FS2 = FunctionSpace(self.mesh,FE_scalar2)
		# Taylor Hodd elements ; stable element pair + eddy viscosity
		self.FS = FunctionSpace(self.mesh,ufl.MixedElement(FE_vector,FE_scalar,FE_scalar2))
		W = FunctionSpace(self.mesh,FE_constant)
		self.FS0c, self.FS_to_FS0 = self.FS.sub(0).collapse()
		self.FS1c, self.FS_to_FS1 = self.FS.sub(1).collapse()
		self.FS2c, self.FS_to_FS2 = self.FS.sub(0).collapse()
		
		# Test & trial functions
		self.trial = ufl.TrialFunction(self.FS)
		self.test  = ufl.TestFunction( self.FS)
		
		# Initialisation of baseflow
		self.Q = Function(self.FS)
		# Collapsed subspaces
		self.U, self.P, self.Nu = Function(self.FS0), Function(self.FS1), Function(self.FS2)
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
			return v[dx].dx(dx) + (r*v[dr]).dx(dr)/r + m*dfx.fem.Constant(self.mesh, 1j)*v[dt]/r
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

	def crl(self,v,m,i=0):
		r=self.r
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		return ufl.as_vector([(i+1)*v[dt]		+r*v[dt].dx(dr)-m*dfx.fem.Constant(self.mesh, 1j)*v[dr],
							   m*1j*v[dx]		-  v[dt].dx(dx),
							  		v[dr].dx(dx)-i*v[dx]-v[dx].dx(dr)])

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

	def Ome(self,u):
		r=self.r
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		a=ufl.sqrt((r*u[dt].dx(dr) + u[dt])**2 + r*u[dt].dx(dx)**2 + (r*(u[dr].dx(dx) - u[dx].dx(dr)))**2)
		return ufl.conditional(ufl.ge(a,self.params['atol']),a,self.params['atol']) # max(a,atol)
	def Omep(self,u,du):
		r=self.r
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		return ((r*du[dt].dx(dr) + du[dt])*(r*u[dt].dx(dr) + u[dt]) + r**2*du[dt].dx(dx)*u[dt].dx(dx) + r**2*(du[dr].dx(dx) - du[dx].dx(dr))*(u[dr].dx(dx) - u[dx].dx(dr)))/self.Ome(u)

	def S(self,q,dist):
		r=self.r
		u, p, nu = ufl.split(q)
		return self.Ome(u) + r*nu*fv2(nu)/(self.Re*kap**2*dist(self)**2)
	def Sp(self,q,dq,dist):
		r=self.r
		u, p, nu = ufl.split(q)
		du, dp, dnu = ufl.split(dq)
		return self.Omep(u,du) + r*(dnu*fv2(nu) + nu*fv2p(nu,dnu))/(self.Re*kap**2*dist(self)**2)

	def ra(self,q,dist):
		u, p, nu = ufl.split(q)
		a=self.r*nu/(self.Re*self.S(q,dist)*kap**2*dist(self)**2)
		return ufl.conditional(ufl.le(a,10),a,10) # min(a,10)
	def rap(self,q,dq,dist):
		r,Re = self.r,self.Re
		u,  p,  nu  = ufl.split(q)
		du, dp, dnu = ufl.split(dq)
		return ((r*nu/(Re*self.S(q,dist)*kap**2*dist(self)**2))<10.)*r*(dnu/(Re*self.S(q,dist)*kap**2*dist(self)**2)
				- nu*self.Sp(q,dq,dist)/(Re*self.S(q,dist)**2*kap**2*dist(self)**2))

	def g(self,q,dist): return self.ra(q,dist) + cw2*(self.ra(q,dist)**6 - self.ra(q,dist))
	def gp(self,q,dq,dist): return (1. + cw2*(6.*self.ra(q,dist)**5 - 1.))*self.rp(q,dq,dist)

	def fw(self,q,dist): return self.g(q,dist)*((1. + cw3**6)/(self.g(q,dist)**6 + cw3**6))**(1./6.)
	def fwp(self,q,dq,dist): return cw3**6*self.gp(q,dq,dist)/(1. + cw3**6)*((1. + cw3**6)/(self.g(q,dist)**6 + cw3**6))**(7./6.)

	# Helper
	def loadBaseflow(self,Re,nut,S,p=False):
		# Load separately
		loadStuff(self.u_path,['S','nut','Re'],[S,nut,Re],self.U)
		loadStuff(self.nut_path,['S','nut','Re'],[S,nut,Re],self.Nu)
		if p: loadStuff(self.p_path,['S','nut','Re'],[S,nut,Re],self.P)
		# Write inside MixedElement
		self.Q.x.array[self.FS_to_FS0]=self.U.x.array
		self.Q.x.array[self.FS_to_FS2]=self.Nu.x.array
		if p:
			self.Q.x.array[self.FS_to_FS1]=self.P.x.array
		self.Q.x.scatter_forward()

	def saveBaseflow(self,str):
		self.Q.x.scatter_forward()
		# Write inside MixedElement
		self.U.x.array[:]=self.Q.x.array[self.FS_to_FS0]
		self.P.x.array[:]=self.Q.x.array[self.FS_to_FS1]
		self.Nu.x.array[:]=self.Q.x.array[self.FS_to_FS2]
		dirCreator(self.u_path)
		dirCreator(self.p_path)
		dirCreator(self.nut_path)
		saveStuff(self.u_path,str,self.U)
		saveStuff(self.p_path,str,self.P)
		saveStuff(self.nut_path,str,self.Nu)
	
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
		self.U.x.array[:]=self.Q.x.array[self.FS_to_FS0]
		# Split arguments
		U,_,Nu = ufl.split(self.Q)
		v,_,_ = ufl.split(self.test)

		# Shorthands
		h,nu=self.h,1/self.Re+Nu
		grd=self.grd

		# Weird Johann local tau (SUPG stabilisation)
		i=ufl.Index()
		n=ufl.sqrt(self.U[i]*self.U[i])
		Pe=ufl.real(n*h/2/nu)
		z=ufl.conditional(ufl.le(Pe,3),Pe/3,1)
		tau=z*h/2/n
		self.SUPG = tau*grd(v,m)*U # Streamline Upwind Petrov Galerkin

	def SA(self, dist):
		# Shortforms
		r, Q, Re = self.r, self.Q, self.Re
		grd=lambda v,i=0: self.grd(v,0,i)
		
		# Functions
		U, _, Nu = ufl.split(Q)
		v, _, t  = ufl.split(self.test)
		# Eddy viscosity term
		F  = ufl.inner(Nu*fv1(Nu)/Re*(grd(U)+grd(U).T),grd(v,1))
		# SA equations
		F += ufl.inner(grd(Nu)*U-cb1*(1. - ft2(Nu))*self.S(Q,dist)*Nu + r*(cw1*self.fw(Q,dist) - cb1/kap**2*ft2(Nu))*Nu**2/(Re*dist(self))**2,t)
		F -= 1./(Re*sig)*ufl.inner((1. + Nu)*grd(Nu),grd(t))
		F += cb2/(Re*sig)*ufl.inner(ufl.inner(grd(Nu),grd(Nu)),t)
		return F*ufl.dx
	
	# Heart of this entire code
	def navierStokes(self,weak_bcs,dist,stabilise=False) -> ufl.Form:
		# Shortforms
		r, Re = self.r, self.Re
		div,grd=lambda v,i=0: self.div(v,0,i), lambda v,i=0: self.grd(v,0,i)
		r2vis, SUPG = lambda v: self.r2vis2(v,0), stabilise*self.SUPG
		
		# Functions
		U, P, _ = ufl.split(self.Q)
		v, s, _ = ufl.split(self.test)
		
		# Mass (variational formulation)
		F  = ufl.inner(div(U),   r*s)
		# Momentum (different test functions and IBP)
		F += ufl.inner(grd(U)*U,r*(v+SUPG)) # Convection
		F -= ufl.inner(	 r*P,  div(v,1)) # Pressure
		F += ufl.inner(grd(U), grd(v,1))/Re
		if stabilise:
			F += ufl.inner(  grd(P),r*SUPG)
			F -= ufl.inner(r2vis(U),  SUPG)
		return F*ufl.dx+weak_bcs(self,U,P)+self.SA(dist)

	def SAlin(self, m, dist):
		# Shortforms
		r, q, Q, Re = self.r, self.trial, self.Q, self.Re
		grd = self.grd
		
		# Functions
		U, _, Nu = ufl.split(Q) # Baseflow
		u, _, nu = ufl.split(q)
		v, _, t  = ufl.split(self.test)
		# Eddy viscosity term
		F  = ufl.inner(nu*fv1(Nu)/Re*(grd(U,0)+grd(U,0).T)+Nu*fv1p(Nu,nu)/Re*(grd(U,0)+grd(U,0).T)+Nu*fv1(Nu)/Re*(grd(u,m)+grd(u,m).T),grd(v,m,1))
		# SA equations
		F += ufl.inner(grd(nu,m)*U+grd(Nu,0)*u,t)
		F += ufl.inner(cb1*ft2p(Nu,nu)*self.S(Q,dist)*Nu-cb1*(1-ft2(Nu))*self.Sp(Q,q,dist)*Nu-cb1*(1-ft2(Nu))*self.S(Q,dist)*nu,t)
		F += r*ufl.inner((cw1*self.fwp(Q,q,dist) - cb1/kap**2*ft2p(Nu,nu))*Nu**2+2*(cw1*self.fw(Q,dist) - cb1/kap**2*ft2(Nu))*nu*Nu,t)/(Re*dist(self))**2
		F -= 1./(Re*sig)*ufl.inner(nu*grd(Nu,0)+(1. + Nu)*grd(nu,m),grd(t,m))
		F += 2*cb2/(Re*sig)*ufl.inner(ufl.inner(grd(Nu,0),grd(nu,m)),t)
		return F*ufl.dx
		
	# Not automatic because of convection term
	def linearisedNavierStokes(self,weak_bcs,m:int,dist,stabilise=False) -> ufl.Form:
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
		return F*ufl.dx+weak_bcs(self,u,p,m)+self.SAlin(m,dist)

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
		self.U.x.array[:]=self.Q.x.array[self.FS_to_FS0]
		self.printStuff("./","sanity_check_u"+app,self.U)

	def sanityCheck(self,app=""):
		self.U.x.array[:]=self.Q.x.array[self.FS_to_FS0]
		self.P.x.array[:]=self.Q.x.array[self.FS_to_FS1]
		self.Nu.x.array[:]=self.Q.x.array[self.FS_to_FS2]

		expr=dfx.fem.Expression(self.div_nor(self.U,0),self.FS1.element.interpolation_points())
		div = Function(self.FS1)
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
		try: self.printStuff("./","sanity_check_nut"+app,self.Nu)
		except TypeError: pass

	def sanityCheckBCs(self):
		v=Function(self.FS)
		v.vector.zeroEntries()
		v.x.array[self.dofs]=np.ones(self.dofs.size)
		v,_=v.split()
		self.printStuff("./","sanity_check_bcs",v)