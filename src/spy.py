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
cw1 = cb1/kap**2 + (1. + cb2)/sig
cw2,cw3 = .3,2
cv1 = 7.1
ct3,ct4 = 1.2,.5

def ft2(nu): 	  return ct3*ufl.exp(-ct4*nu**2)
def ft2p(nu,dnu): return -2.*ct4*ct3*nu*dnu*ufl.exp(-ct4*nu**2)

def fv1(c): 	return c**3/(c**3 + cv1**3)
def fv1p(c,dc): return 3.*cv1**3*c**2*dc/(c**3 + cv1**3)**2

def fv2(c): 	return 1. - c/(1. + c*fv1(c))
def fv2p(c,dc): return (c**2*fv1p(c,dc) - dc)/(1. + c*fv1(c))**2

def Ome(u,r,dx:int,dr:int,dt:int,atol:float,mesh):
	c=crl(r,dx,dr,dt,mesh,u,0)
	return ufl.sqrt(.5*ufl.dot(c,c)+dfx.fem.Constant(mesh,pet.ScalarType(atol)))
	a=ufl.sqrt((r*u[dt].dx(dr) + u[dt])**2 + r*u[dt].dx(dx)**2 + (r*(u[dr].dx(dx) - u[dx].dx(dr)))**2)
	return ufl.conditional(ufl.ge(a,atol),a,atol) # max(a,atol)
def Omep(u,du,r,dx:int,dr:int,dt:int,atol:float,mesh,m:int):
	c =crl(r,dx,dr,dt,mesh,u,0)
	dc=crl(r,dx,dr,dt,mesh,du,m)
	return ufl.dot(c,dc)/Ome(u,r,dx,dr,dt,atol,mesh)
	return ((r*du[dt].dx(dr) + du[dt])*(r*u[dt].dx(dr) + u[dt]) + r**2*du[dt].dx(dx)*u[dt].dx(dx) + r**2*(du[dr].dx(dx) - du[dx].dx(dr))*(u[dr].dx(dx) - u[dx].dx(dr)))/Ome(u,r,dx,dr,dt,atol)

def S(q,r,dx:int,dr:int,dt:int,atol:float,Re,d,mesh):
	u, _, nu = ufl.split(q)
	return Ome(u,r,dx,dr,dt,atol,mesh) + r*nu*fv2(nu*Re)/(kap*d)**2
def Sp(q,dq,r,dx:int,dr:int,dt:int,atol:float,Re,d,m):
	u,  _, nu  = ufl.split(q)
	du, _, dnu = ufl.split(dq)
	return Omep(u,du,r,dx,dr,dt,atol,m) + r*(dnu*fv2(nu*Re) + nu*fv2p(nu*Re,dnu*Re))/(kap*d)**2

def ra(q,r,dx:int,dr:int,dt:int,atol:float,d,mesh):
	u, _, nu = ufl.split(q)
	return r*nu/Ome(u,r,dx,dr,dt,atol,mesh)/(kap*d)**2
	#return ufl.conditional(ufl.le(a,10),a,10) # min(a,10)
def rap(q,dq,r,dx:int,dr:int,dt:int,atol:float,Re,d):
	_, _,  nu = ufl.split(q)
	_, _, dnu = ufl.split(dq)
	Sv=S(q,r,dx,dr,dt,atol,Re,d)
	return ((r*nu/(Re*Sv*kap**2*d**2))<10.)*r*(dnu/(Re*Sv*kap**2*d**2)
			- nu*Sp(q,dq,r,dx,dr,dt,atol,Re,d)/(Re*Sv**2*kap**2*d**2))

def g(q,r,dx:int,dr:int,dt:int,atol:float,Re,d):
	rav=ra(q,r,dx,dr,dt,atol,Re,d)
	return rav + cw2*(rav**6 - rav)
def gp(q,dq,r,dx:int,dr:int,dt:int,atol:float,Re,d): return (1. + cw2*(6.*ra(q,r,dx,dr,dt,atol,Re,d)**5 - 1.))*rap(q,dq,r,dx,dr,dt,atol,Re,d)

def fw(q,r,dx:int,dr:int,dt:int,atol:float,Re,d):
	gv=g(q,r,dx,dr,dt,atol,Re,d)
	return gv*((1 + cw3**6)/(gv**6 + cw3**6))**(1./6.)
def fwp(q,dq,r,dx:int,dr:int,dt:int,atol:float,Re,d): return cw3**6*gp(q,dq,r,dx,dr,dt,atol,Re,d)/(1. + cw3**6)*((1. + cw3**6)/(g(q,r,dx,dr,dt,atol,Re,d)**6 + cw3**6))**(7./6.)

# Vanilla operators
def grd_nor(r,dx:int,dr:int,dt:int,v,m:int):
	if len(v.ufl_shape)==0:
		return ufl.as_vector([v.dx(dx), v.dx(dr), m*1j*v/r])
	return ufl.as_tensor([[v[dx].dx(dx), v[dx].dx(dr),  m*1j*v[dx]		 /r],
							[v[dr].dx(dx), v[dr].dx(dr), (m*1j*v[dr]-v[dt])/r],
							[v[dt].dx(dx), v[dt].dx(dr), (m*1j*v[dt]+v[dr])/r]])

def div_nor(r,dx:int,dr:int,dt:int,mesh,v,m:int):
	if len(v.ufl_shape)==1:
		return v[dx].dx(dx) + (r*v[dr]).dx(dr)/r + m*dfx.fem.Constant(mesh, 1j)*v[dt]/r
	else:
		return ufl.as_vector([v[dx,dx].dx(dx)+v[dr,dx].dx(dr)+(v[dr,dx]+m*1j*v[dt,dx])/r,
								v[dx,dr].dx(dx)+v[dr,dr].dx(dr)+(v[dr,dr]+m*1j*v[dt,dr]-v[dt,dt])/r,
								v[dx,dt].dx(dx)+v[dr,dt].dx(dr)+(v[dr,dt]+m*1j*v[dt,dt]+v[dt,dr])/r])

# Gradient with r multiplication
def grd(r,dx:int,dr:int,dt:int,				 v,m:int,i:int=0):
	if len(v.ufl_shape)==0: return ufl.as_vector([r*v.dx(dx), i*v+r*v.dx(dr), m*1j*v])
	return ufl.as_tensor([[r*v[dx].dx(dx), i*v[dx]+r*v[dx].dx(dr), m*1j*v[dx]],
							[r*v[dr].dx(dx), i*v[dr]+r*v[dr].dx(dr), m*1j*v[dr]-v[dt]],
							[r*v[dt].dx(dx), i*v[dt]+r*v[dt].dx(dr), m*1j*v[dt]+v[dr]]])

def div(r,dx:int,dr:int,dt:int,				 v,m:int,i:int=0):
	return r*v[dx].dx(dx) + (1+i)*v[dr] + r*v[dr].dx(dr) + m*1j*v[dt]

def crl(r,dx:int,dr:int,dt:int,mesh:ufl.Mesh,v,m:int,i:int=0):
	return ufl.as_vector([(i+1)*v[dt]		+r*v[dt].dx(dr)-m*dfx.fem.Constant(mesh, 1j)*v[dr],
   m*dfx.fem.Constant(mesh, 1j)*v[dx]		-  v[dt].dx(dx),
								v[dr].dx(dx)-i*v[dx]-v[dx].dx(dr)])

def r2vis( r,dx:int,dr:int,dt:int,nu,v,m:int):
	return ufl.as_vector([2*r**2*(nu*v[dx].dx(dx)).dx(dx)			  +r*(r*nu*(v[dr].dx(dx)+v[dx].dx(dr))).dx(dr)+m*1j*nu*(r*v[dt].dx(dx)+m*1j*v[dx]),
							r**2*(nu*(v[dr].dx(dx)+v[dx].dx(dr))).dx(dx)+r*(2*r*nu*v[dr].dx(dr)).dx(dr)			  +m*1j*nu*(r*v[dt].dx(dr)+m*1j*v[dr])-2*m*1j*nu*v[dt],
							r  *(nu*(r*v[dt].dx(dx)+m*1j*v[dx])).dx(dx) +r*(nu*(r*v[dt].dx(dr)+m*1j*v[dr])).dx(dr)-2*m**2*nu*v[dt]					  +nu*(r*v[dt].dx(dr)+m*1j*v[dr])])

def r2vis2(r,dx:int,dr:int,dt:int,nu,v,m:int):
	return ufl.as_vector([r**2*(nu*v[dx].dx(dx)).dx(dx)+r*(r*nu*v[dr].dx(dx)).dx(dr)+m*1j*nu*r*v[dt].dx(dx),
							r**2*(nu*v[dx].dx(dr)).dx(dx)+r*(r*nu*v[dr].dx(dr)).dx(dr)+m*1j*nu*r*v[dt].dx(dr)-m*1j*nu*v[dt],
							r   *m*1j*(nu*v[dx]).dx(dx)  +r*m*1j*(nu*v[dr]).dx(dr)	-m**2*nu*v[dt]		   +m*1j*nu*v[dr]])

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

		self.defineFunctionSpaces()
	
		# Forcing localisation
		if forcingIndicator==None: self.indic=1
		else:
			FE_constant=ufl.FiniteElement("DG",self.mesh.ufl_cell(),0)
			W = FunctionSpace(self.mesh,FE_constant)
			self.indic = Function(W)
			self.indic.interpolate(forcingIndicator)

		# BCs essentials
		self.dofs = np.empty(0,dtype=np.int32)
		self.bcs  = []

	def defineFunctionSpaces(self):
		# Extraction of r
		self.r = ufl.SpatialCoordinate(self.mesh)[self.direction_map['r']]
		# Finite elements & function spaces
		FE_vector=ufl.VectorElement("CG",self.mesh.ufl_cell(),2,3)
		FE_scalar=ufl.FiniteElement("CG",self.mesh.ufl_cell(),1)
		self.FS0 = FunctionSpace(self.mesh,FE_vector)
		self.FS1 = FunctionSpace(self.mesh,FE_scalar)
		self.FS2 = FunctionSpace(self.mesh,FE_scalar)
		# Taylor Hodd elements ; stable element pair + eddy viscosity
		self.FS = FunctionSpace(self.mesh,ufl.MixedElement(FE_vector,FE_scalar,FE_scalar))
		self.FS0c, self.FS_to_FS0 = self.FS.sub(0).collapse()
		self.FS1c, self.FS_to_FS1 = self.FS.sub(1).collapse()
		self.FS2c, self.FS_to_FS2 = self.FS.sub(2).collapse()
		# Test & trial functions
		self.trial = ufl.TrialFunction(self.FS)
		self.test  = ufl.TestFunction( self.FS)
		# Initialisation of baseflow
		self.Q = Function(self.FS)
		# Collapsed subspaces
		self.U, self.P, self.Nu = Function(self.FS0), Function(self.FS1), Function(self.FS2)

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
		self.U.x.array[:] =self.Q.x.array[self.FS_to_FS0]
		self.P.x.array[:] =self.Q.x.array[self.FS_to_FS1]
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
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		r=self.r
		gd=lambda v,i=0: grd(r,dx,dr,dt,v,0,i)
		a=ufl.inner(u,v)*r**2
		a+=e*ufl.inner(gd(u),gd(v,1))
		L=ufl.inner(fun,v)
		pb = dfx.fem.petsc.LinearProblem(a*ufl.dx+weak_bcs(self,u,v), L*r**2*ufl.dx, bcs=bcs, petsc_options={"ksp_type": "cg", "pc_type": "gamg", "pc_factor_mat_solver_type": "mumps"})
		if p0: print("Smoothing started...",flush=True)
		res=pb.solve()
		res.x.scatter_forward()
		return res.x.array

	def stabilise(self,m):
		# Split arguments
		U,_,Nu = ufl.split(self.Q)
		v,_,_  = ufl.split(self.test)
		# Important ! Otherwise n is nonsense
		self.U.x.array[:]=self.Q.x.array[self.FS_to_FS0]
		# Shorthands
		nu=1/self.Re+Nu
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		gdv=grd(self.r,dx,dr,dt,v,m)
		# Local mesh size
		h = ufl.CellDiameter(self.mesh)
		# Weird Johann local tau (SUPG stabilisation)
		i=ufl.Index()
		n=ufl.sqrt(self.U[i]*self.U[i])
		Pe=ufl.real(n*h/2/nu)
		z=ufl.conditional(ufl.le(Pe,3),Pe/3,1)
		tau=z*h/2/n
		self.SUPG = tau*gdv*U # Streamline Upwind Petrov Galerkin

	def SA(self, d):
		# Shortforms
		r, Q, Re = self.r, self.Q, self.Re
		# Functions
		U, _, Nu = ufl.split(Q)
		v, _, t  = ufl.split(self.test)
		# More shortforms
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		r,atol=self.r,self.params['atol']
		gd=lambda v,i=0: grd(r,dx,dr,dt,v,0,i)
		fwv,Sv=fw(Q,r,dx,dr,dt,atol,Re,d),S(Q,r,dx,dr,dt,atol,Re,d,self.mesh)
		# Eddy viscosity term
		F  = ufl.inner(Nu*fv1(Nu)/Re*(gd(U)+gd(U).T),gd(v,1))
		# SA equations
		F += ufl.inner(ufl.dot(gd(Nu),U),t)
		F -= cb1*ufl.inner(Sv*Nu,t)
		F += 1./Re/sig*ufl.inner((1. + Nu)*gd(Nu),gd(t,1))
		F -= cb2/Re/sig*ufl.inner(ufl.dot(gd(Nu),gd(Nu)),t)
		F += cw1*ufl.inner(fwv*(Nu/d)**2,t)
		return F*ufl.dx
	
	# Heart of this entire code
	def navierStokes(self,weak_bcs,dist,stabilise=False) -> ufl.Form:
		# Shortforms
		r, Re = self.r, self.Re
		# Functions
		U, P, _ = ufl.split(self.Q)
		v, s, _ = ufl.split(self.test)
		# More shortforms
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		dv,gd=lambda v,i=0: div(r,dx,dr,dt,v,0,i),lambda v,i=0: grd(r,dx,dr,dt,v,0,i)
		r2vs, SUPG = r2vis2(r,dx,dr,dt,1./Re,U,0), stabilise*self.SUPG
		# Mass (variational formulation)
		F  = ufl.inner(dv(U),   r*s)
		# Momentum (different test functions and IBP)
		F += ufl.inner(gd(U)*U,r*(v+SUPG)) # Convection
		F -= ufl.inner(	r*P,   dv(v,1)) # Pressure
		F += ufl.inner(gd(U),  gd(v,1))/Re
		if stabilise:
			F += ufl.inner(gd(P),r*SUPG)
			F -= ufl.inner(r2vs,   SUPG)
		return F*ufl.dx+weak_bcs(self,U,P)+self.SA(dist)

	def SAlin(self, m, d):
		# Shortforms
		r, q, Q, Re, atol = self.r, self.trial, self.Q, self.Re, self.params['atol']
		# Functions
		U, _, Nu = ufl.split(Q) # Baseflow
		u, _, nu = ufl.split(q)
		v, _, t  = ufl.split(self.test)
		# More shortforms
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		gd=lambda v,m,i=0: grd(r,dx,dr,dt,v,m,i)
		Sv,Spv=S(Q,r,dx,dr,dt,atol,Re,d),Sp(Q,q,r,dx,dr,dt,atol,Re,d)
		fwv,fwpv=fw(Q,r,dx,dr,dt,atol,Re,d),fwp(Q,q,r,dx,dr,dt,atol,Re,d)
		# Eddy viscosity term
		F  = ufl.inner(nu*fv1(Nu)/Re*(gd(U,0)+gd(U,0).T)
				  +Nu*fv1p(Nu,nu)/Re*(gd(U,0)+gd(U,0).T)
				      +Nu*fv1(Nu)/Re*(gd(u,m)+gd(u,m).T),gd(v,m,1))
		# SA equations
		F += ufl.inner(ufl.dot(gd(nu,m),U)+ufl.dot(gd(Nu,0),u),t)
		F += ufl.inner(cb1*ft2p(Nu,nu)*Sv*Nu-cb1*(1-ft2(Nu))*Spv*Nu-cb1*(1-ft2(Nu))*Sv*nu,t)
		F += r*ufl.inner((cw1*fwpv - cb1/kap**2*ft2p(Nu,nu))*Nu**2+2*(cw1*fwv - cb1/kap**2*ft2(Nu))*nu*Nu,t)/(Re*d)**2
		F -= 1./(Re*sig)*ufl.inner(nu*gd(Nu,0)+(1. + Nu)*gd(nu,m),gd(t,m,1))
		F += 2*cb2/(Re*sig)*ufl.inner(ufl.dot(gd(Nu,0),gd(nu,m)),t)
		return F*ufl.dx
		
	# Not automatic because of convection term
	def linearisedNavierStokes(self,weak_bcs,m:int,dist,stabilise=False) -> ufl.Form:
		# Shortforms
		r,Re=self.r,self.Re
		# Functions
		u, p, _ = ufl.split(self.trial)
		U, _, _ = ufl.split(self.Q) # Baseflow
		v, s, _ = ufl.split(self.test)
		# More shortforms
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		dv,gd=lambda v,m,i=0: div(r,dx,dr,dt,v,m,i),lambda v,m,i=0: grd(r,dx,dr,dt,v,m,i)
		r2vs, SUPG = r2vis2(r,dx,dr,dt,1./Re,u,m), stabilise*self.SUPG
		# Mass (variational formulation)
		F  = ufl.inner(dv(u,m),  r*s)
		# Momentum (different test functions and IBP)
		F += ufl.inner(gd(U,0)*u,r*(v+SUPG)) # Convection
		F += ufl.inner(gd(u,m)*U,r*(v+SUPG))
		F -= ufl.inner(  r*p,    dv(v,m,1)) # Pressure
		#F += ufl.inner(nu*(grd(u,m)+grd(u,m).T),grd(v,m,1)) # Diffusion (grad u.T significant with nut)
		F += ufl.inner(gd(u,m),gd(v,m,1))/Re
		if stabilise:
			F += ufl.inner(gd(p,m),r*SUPG)
			F -= ufl.inner(r2vs,	 SUPG)
		#F += ufl.inner(div(u,m),self.grd_div)
		return F*ufl.dx+weak_bcs(self,u,p,m)+self.SAlin(m,dist)

	# Code factorisation
	def constantBC(self, direction:chr, boundary, value=0, subspace=0) -> tuple:
		sub_space=self.FS.sub(subspace).sub(self.direction_map[direction])
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

	def applyHomogeneousBCs(self, tup:list, subspace=0) -> None:
		for marker,directions in tup:
			for direction in directions:
				dofs,bcs=self.constantBC(direction,marker, subspace)
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

		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		expr=dfx.fem.Expression(div_nor(self.r,dx,dr,dt,self.mesh,self.U,0),self.FS1.element.interpolation_points())
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