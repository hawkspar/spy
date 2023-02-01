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
sig = 2/3
kap = .41
cw1 = cb1/kap**2 + (1+cb2)/sig
cw2,cw3 = .3,2
cv1 = 7.1
ct3,ct4 = 1.2,.5

def ft2(C):    return ct3*ufl.exp(-ct4*C**2)
def ft2p(C,c): return -2*ct4*ct3*C*c*ufl.exp(-ct4*C**2)

def fv1(C):    return C**3/(C**3 + cv1**3)
def fv1p(C,c): return 3*cv1**3*C**2*c/(C**3 + cv1**3)**2

def fv2(C):    return 1 - C/(1 + C*fv1(C))
def fv2p(C,c): return (C**2*fv1p(C,c) - c)/(1 + C*fv1(C))**2

def Ome(W,atol): 	return ufl.sqrt(.5*ufl.dot(W,W)+atol)
def Omep(W,w,atol): return .5*ufl.dot(W,w)/Ome(W,atol)

# Actually S*r
def S( Nu,	 r,W,  atol,Re,ikd2): return Ome(W,atol) 	+ r* Nu*fv2(Nu*Re)*ikd2
def Sp(Nu,nu,r,W,w,atol,Re,ikd2): return Omep(W,w,atol) + r*(nu*fv2(Nu*Re) + Nu*fv2p(Nu*Re,nu*Re))*ikd2

def ra(Nu,r,W,atol,Re,ikd2):
	a = r*Nu/S(Nu,r,W,atol,Re,ikd2)*ikd2 # r to cancel out the multiplication
	return ufl.conditional(ufl.le(a,10),a,10) # min(a,10)
def rap(Nu,nu,r,W,w,atol,Re,ikd2):
	Sv = S(Nu,r,W,atol,Re,ikd2)
	a = r*Nu/Sv*ikd2
	return ufl.conditional(ufl.le(a,10),nu-Nu*Sp(Nu,nu,r,W,w,atol,Re,ikd2)/Sv,0)*r/Sv*ikd2

def g(Nu,r,W,atol,Re,ikd2):
	rav=ra(Nu,r,W,atol,Re,ikd2)
	return rav + cw2*(rav**6 - rav)
def gp(Nu,nu,r,W,w,atol,Re,ikd2): return (1 + cw2*(6*ra(Nu,r,W,atol,Re,ikd2)**5 - 1))*rap(Nu,nu,r,W,w,atol,Re,ikd2)

def fw(Nu,r,W,atol,Re,ikd2):
	gv=g(Nu,r,W,atol,Re,ikd2)
	return gv*((1 + cw3**6)/(gv**6 + cw3**6))**(1/6)
def fwp(Nu,nu,r,W,w,atol,Re,ikd2): return cw3**6*(1 + cw3**6)**(1/6)*gp(Nu,nu,r,W,w,atol,Re,ikd2)/(g(Nu,r,W,atol,Re,ikd2)**6 + cw3**6)**(7/6)

# Vanilla operators
def grd_nor(r,dx:int,dr:int,dt:int,v,m:int):
	if len(v.ufl_shape)==0: return ufl.as_vector([v.dx(dx), v.dx(dr), m*1j*v/r])
	return ufl.as_tensor([[v[dx].dx(dx), v[dx].dx(dr),  m*1j*v[dx]		 /r],
						  [v[dr].dx(dx), v[dr].dx(dr), (m*1j*v[dr]-v[dt])/r],
						  [v[dt].dx(dx), v[dt].dx(dr), (m*1j*v[dt]+v[dr])/r]])

def div_nor(r,dx:int,dr:int,dt:int,mesh,v,m:int):
	if len(v.ufl_shape)==1: return v[dx].dx(dx) + (r*v[dr]).dx(dr)/r + m*dfx.fem.Constant(mesh, 1j)*v[dt]/r
	return ufl.as_vector([v[dx,dx].dx(dx)+v[dr,dx].dx(dr)+(v[dr,dx]+m*1j*v[dt,dx])/r,
						  v[dx,dr].dx(dx)+v[dr,dr].dx(dr)+(v[dr,dr]+m*1j*v[dt,dr]-v[dt,dt])/r,
						  v[dx,dt].dx(dx)+v[dr,dt].dx(dr)+(v[dr,dt]+m*1j*v[dt,dt]+v[dt,dr])/r])

# Operators with r multiplication
def grd(r,dx:int,dr:int,dt:int,				 v,m:int,i:int=0):
	if len(v.ufl_shape)==0: return ufl.as_vector([r*v.dx(dx), i*v+r*v.dx(dr), m*1j*v])
	return ufl.as_tensor([[r*v[dx].dx(dx), i*v[dx]+r*v[dx].dx(dr), m*1j*v[dx]],
						  [r*v[dr].dx(dx), i*v[dr]+r*v[dr].dx(dr), m*1j*v[dr]-v[dt]],
						  [r*v[dt].dx(dx), i*v[dt]+r*v[dt].dx(dr), m*1j*v[dt]+v[dr]]])

def div(r,dx:int,dr:int,dt:int,				 v,m:int,i:int=0):
	if len(v.ufl_shape)==1: return r*v[dx].dx(dx) + (1+i)*v[dr] + v[dr].dx(dr) + m*1j*v[dt]
	return ufl.as_vector([r*v[dx,dx].dx(dx)+(1+i)*v[dr,dx].dx(dr)+v[dr,dx]+m*1j*v[dt,dx],
						  r*v[dx,dr].dx(dx)+(1+i)*v[dr,dr].dx(dr)+v[dr,dr]+m*1j*v[dt,dr]-v[dt,dt],
						  r*v[dx,dt].dx(dx)+(1+i)*v[dr,dt].dx(dr)+v[dr,dt]+m*1j*v[dt,dt]+v[dt,dr]])

def crl(r,dx:int,dr:int,dt:int,mesh:ufl.Mesh,v,m:int,i:int=0):
	return ufl.as_vector([(i+1)*v[dt]		+r*v[dt].dx(dr)-m*dfx.fem.Constant(mesh, 1j)*v[dr],
    m*dfx.fem.Constant(mesh,1j)*v[dx]		-  v[dt].dx(dx),
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
def meshConvert(path:str,cell_type:str='triangle',prune=True) -> None:
	import meshio #pip3 install --no-binary=h5py h5py meshio
	gmsh_mesh = meshio.read(path+".msh")
	# Write it out again
	ps = gmsh_mesh.points[:,:(3-prune)]
	cs = gmsh_mesh.get_cells_type(cell_type)
	dolfinx_mesh = meshio.Mesh(points=ps, cells={cell_type: cs})
	meshio.write(path+".xdmf", dolfinx_mesh)
	print("Mesh "+path+".msh converted to "+path+".xdmf !",flush=True)

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
	def __init__(self, params:dict, datapath:str, mesh_name:str, direction_map:dict, forcingIndicator=None) -> None:
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
		meshpath=self.case_path+mesh_name+".xdmf"
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

	# TO be rerun if mesh changes
	def defineFunctionSpaces(self):
		# Extraction of r
		self.r = ufl.SpatialCoordinate(self.mesh)[self.direction_map['r']]
		# Finite elements & function spaces
		FE_vector =ufl.VectorElement("CG",self.mesh.ufl_cell(),2,3)
		FE_scalar =ufl.FiniteElement("CG",self.mesh.ufl_cell(),1)
		FE_scalar2=ufl.FiniteElement("CG",self.mesh.ufl_cell(),2)
		#Constant =ufl.FiniteElement("Real",self.mesh.ufl_cell(),0)
		self.FS0 = FunctionSpace(self.mesh,FE_vector)
		self.FS1 = FunctionSpace(self.mesh,FE_scalar)
		self.FS2 = FunctionSpace(self.mesh,FE_scalar2)
		# Taylor Hodd elements ; stable element pair + eddy viscosity
		self.FS = FunctionSpace(self.mesh,ufl.MixedElement(FE_vector,FE_scalar))#,FE_scalar2))
		self.FS0c, self.FS_to_FS0 = self.FS.sub(0).collapse()
		self.FS1c, self.FS_to_FS1 = self.FS.sub(1).collapse()
		#self.FS2c, self.FS_to_FS2 = self.FS.sub(2).collapse()
		"""# Extended element for the corrector
		self.FSe = FunctionSpace(self.mesh,ufl.MixedElement(FE_vector,FE_scalar,FE_scalar,Constant))
		self.FSe0c, self.FSe_to_FSe0 = self.FSe.sub(0).collapse()
		self.FSe1c, self.FSe_to_FSe1 = self.FSe.sub(1).collapse()
		self.FSe2c, self.FSe_to_FSe2 = self.FSe.sub(2).collapse()
		self.FSe3c, self.FSe_to_FSe3 = self.FSe.sub(3).collapse()"""
		# Test & trial functions
		self.trial = ufl.TrialFunction(self.FS)
		self.test  = ufl.TestFunction( self.FS)
		"""self.triale = ufl.split(ufl.TrialFunction(self.FSe))
		self.teste  = ufl.split(ufl.TestFunction( self.FSe))"""
		# Initialisation of baseflow
		self.Q = Function(self.FS)
		# Collapsed subspaces
		self.U, self.P, self.Nu = Function(self.FS0), Function(self.FS1), Function(self.FS2)

	def extend(self, Q:Function) -> Function:
		Qe = Function(self.FSe)
		Qe.x.array[self.FSe_to_FSe0]=Q.x.array[self.FS_to_FS0]
		Qe.x.array[self.FSe_to_FSe1]=Q.x.array[self.FS_to_FS1]
		Qe.x.array[self.FSe_to_FSe2]=Q.x.array[self.FS_to_FS2]
		return Qe

	def revert(self, Qe:Function) -> Function:
		Q = Function(self.FS)
		Q.x.array[self.FS_to_FS0]=Qe.x.array[self.FSe_to_FSe0]
		Q.x.array[self.FS_to_FS1]=Qe.x.array[self.FSe_to_FSe1]
		Q.x.array[self.FS_to_FS2]=Qe.x.array[self.FSe_to_FSe2]
		Re=np.mean(Qe.x.array[self.FSe_to_FSe3])
		return Q,Re

	# Helper
	def loadBaseflow(self,Re:int,nut:int,S:float,p=False):
		# Load separately
		loadStuff(self.u_path,['S','nut','Re'],[S,nut,Re],self.U)
		loadStuff(self.nut_path,['S','nut','Re'],[S,nut,Re],self.Nu)
		if p: loadStuff(self.p_path,['S','nut','Re'],[S,nut,Re],self.P)
		# Write inside MixedElement
		self.Q.x.array[self.FS_to_FS0]=self.U.x.array
		#self.Q.x.array[self.FS_to_FS2]=self.Nu.x.array
		if p:
			self.Q.x.array[self.FS_to_FS1]=self.P.x.array
		self.Q.x.scatter_forward()
		self.Nu.x.scatter_forward()

	def saveBaseflow(self,str):
		self.Q.x.scatter_forward()
		# Write inside MixedElement
		self.U.x.array[:] =self.Q.x.array[self.FS_to_FS0]
		self.P.x.array[:] =self.Q.x.array[self.FS_to_FS1]
		#self.Nu.x.array[:]=self.Q.x.array[self.FS_to_FS2]
		dirCreator(self.u_path)
		dirCreator(self.p_path)
		dirCreator(self.nut_path)
		saveStuff(self.u_path,'u'+str,self.U)
		saveStuff(self.p_path,'p'+str,self.P)
		if type(self.Nu)==Function: saveStuff(self.nut_path,'nut'+str,self.Nu)
	
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
		U,_ = ufl.split(self.Q)
		v,_ = ufl.split(self.test)
		Nu=self.Nu
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

	def SA(self, U, Nu, v, t, d):
		# Shortforms
		r, Re = self.r, self.Re
		# More shortforms
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		r,atol=self.r,dfx.fem.Constant(self.mesh,pet.ScalarType(self.params['atol']))
		gd=lambda v,i=0: grd(r,dx,dr,dt,v,0,i)
		W=crl(r,dx,dr,dt,self.mesh,U,0)
		fwv,Sv=fw(Nu,r,W,atol,Re,1/(kap*d)**2),S(Nu,r,W,atol,Re,1/(kap*d)**2)
		# Eddy viscosity term
		F  = ufl.inner(Nu*fv1(Nu*Re)*(gd(U)+gd(U).T),gd(v,1))*ufl.dx
		# SA equations
		G  = r*ufl.inner(ufl.dot(gd(Nu),U),t)
		G -= r*cb1*(1-ft2(Nu*Re))*ufl.inner(Sv*Nu,t)
		G += r**2*ufl.inner(cw1*fwv-cb1/kap**2*ft2(Nu*Re),t)*(Nu/d)**2
		G += 1/sig*ufl.inner((1/Re+Nu)*gd(Nu),gd(t,1))
		G -= cb2/sig*ufl.inner(ufl.dot(gd(Nu),gd(Nu)),t)
		return F+G*ufl.dx(degree=30)
	
	# Heart of this entire code
	def navierStokes(self,Q,Qt,dist,stabilise=False,extended=False) -> ufl.Form:
		# Shortforms
		r, Re = self.r, self.Re
		# Functions
		if extended:
			U, P, Nu, _ = ufl.split(Q)
			v, s, t,  _ = ufl.split(Qt)
		else:
			U, P = ufl.split(Q)
			Nu = self.Nu
			v, s  = ufl.split(Qt)
		# More shortforms
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		dv,gd=lambda v,i=0: div(r,dx,dr,dt,v,0,i),lambda v,i=0: grd(r,dx,dr,dt,v,0,i)
		r2vs, SUPG = r2vis2(r,dx,dr,dt,1./Re,U,0), stabilise*self.SUPG
		# Mass (variational formulation)
		F  = ufl.inner(dv(U),   r*s)
		# Momentum (different test functions and IBP)
		F += ufl.inner(gd(U)*U, r*v)#+SUPG)) # Convection
		F -= ufl.inner(	r*P,   dv(v,1)) # Pressure
		F += ufl.inner((1/Re+Nu)*(gd(U)+gd(U).T),gd(v,1)) # Diffusion (grad u.T significant with nut)
		#F += ufl.inner((1/Re+Nu)*gd(U),gd(v,1))
		#F += ufl.inner(gd(U),  gd(v,1))/Re
		if stabilise:
			F += ufl.inner(gd(P),r*SUPG)
			F -= ufl.inner(r2vs,   SUPG)
		return F*ufl.dx#+self.SA(U,Nu,v,t,dist)
	
	# Heart of this entire code
	def navierStokesError(self) -> ufl.Form:
		# Shortforms
		r, Re = self.r, self.Re
		# Functions
		U, P, Nu = ufl.split(self.Q)
		# More shortforms
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		dv,gd=lambda v,i=0: div(r,dx,dr,dt,v,0,i),lambda v,i=0: grd(r,dx,dr,dt,v,0,i)
		r2vs = r2vis2(r,dx,dr,dt,1/Re+Nu*fv1(Nu),U,0)
		# Mass (variational formulation)
		dv2 = ufl.inner(dv(U), dv(U))
		# Momentum (different test functions and IBP)
		mo  = r*gd(U)*U+r*gd(P)-r2vs
		mo2 = ufl.inner(mo,    mo)
		return (dv2+mo2)*ufl.dx

	def SAlin(self, U, Nu, u, nu, v, t, m, d):
		# Shortforms
		r, Re, atol = self.r, self.Re, self.params['atol']
		# More shortforms
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		r,atol=self.r,dfx.fem.Constant(self.mesh,pet.ScalarType(self.params['atol']))
		gd=lambda v,m,i=0: grd(r,dx,dr,dt,v,m,i)
		W,w=crl(r,dx,dr,dt,self.mesh,U,0),crl(r,dx,dr,dt,self.mesh,u,m)
		Sv, Spv = S(Nu,r,W,atol,Re,1/(kap*d)**2), Sp(Nu,nu,r,W,w,atol,Re,1/(kap*d)**2)
		fwv,fwpv=fw(Nu,r,W,atol,Re,1/(kap*d)**2),fwp(Nu,nu,r,W,w,atol,Re,1/(kap*d)**2)
		# Eddy viscosity term
		F  = ufl.inner((nu*fv1(Nu*Re)
				   +Nu*fv1p(Nu*Re,nu*Re))*(gd(U,0)+gd(U,0).T)
				       +Nu*fv1(Nu*Re)    *(gd(u,m)+gd(u,m).T),gd(v,m,1))*ufl.dx
		# SA equations
		G  = r*ufl.inner(ufl.dot(gd(nu,m),U)+ufl.dot(gd(Nu,0),u),t)
		G -= r*cb1*(ft2p(Nu*Re,nu*Re)*ufl.inner(Sv*Nu,t)+(1-ft2(Nu*Re))*ufl.inner(Spv*Nu+Sv*nu,t))
		G += (r/d)**2*(ufl.inner(cw1*fwpv-cb1/kap**2*ft2p(Nu*Re,nu*Re),t)*Nu**2+2*ufl.inner(cw1*fwv-cb1/kap**2*ft2(Nu*Re),t)*nu*Nu)
		G += 1/sig*ufl.inner(nu*gd(Nu,0)+(1/Re+Nu)*gd(nu,m),gd(t,m,1))
		G -= cb2/sig*ufl.inner(2*ufl.dot(gd(nu,m),gd(Nu,0)),t)
		return F+G*ufl.dx(degree=30)
		
	# Not automatic because of convection term
	def linearisedNavierStokes(self,q,Q,Qt,m:int,dist,stabilise=False,extended=False) -> ufl.Form:
		# Shortforms
		r,Re=self.r,self.Re
		# Functions
		if extended:
			u, p, nu, _ = ufl.split(q)
			U, _, Nu, _ = ufl.split(Q) # Baseflow
			v, s, t,  _ = ufl.split(Qt)
		else:
			u, p = ufl.split(q)
			U, _ = ufl.split(Q) # Baseflow
			Nu = self.Nu 
			v, s = ufl.split(Qt)
		# More shortforms
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		dv,gd=lambda v,m,i=0: div(r,dx,dr,dt,v,m,i),lambda v,m,i=0: grd(r,dx,dr,dt,v,m,i)
		r2vs, SUPG = r2vis2(r,dx,dr,dt,1./Re,u,m), stabilise*self.SUPG
		# Mass (variational formulation)
		F  = ufl.inner(dv(u,m),   r*s)
		# Momentum (different test functions and IBP)
		F += ufl.inner(gd(U,0)*u, r*v)#+SUPG)) # Convection
		F += ufl.inner(gd(u,m)*U, r*v)#+SUPG))
		F -= ufl.inner(  r*p,    dv(v,m,1)) # Pressure
		F += ufl.inner((1/Re+Nu)*(gd(u,m)+gd(u,m).T),gd(v,m,1)) # Diffusion (grad u.T significant with nut)
		#F += ufl.inner((1/Re+Nu)*gd(u,m),gd(v,m,1))
		#F += ufl.inner(gd(u,m),gd(v,m,1))/Re
		if stabilise:
			F += ufl.inner(gd(p,m),r*SUPG)
			F -= ufl.inner(r2vs,	 SUPG)
		#F += ufl.inner(div(u,m),self.grd_div)
		return F*ufl.dx#+self.SAlin(U,Nu,u,nu,v,t,m,dist)

	# Code factorisation
	def constantBC(self, direction:chr, boundary:bool, value:float=0, subspace_i:int=0) -> tuple:
		subspace=self.FS.sub(subspace_i)
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
		self.U.x.array[:]=self.Q.x.array[self.FS_to_FS0]
		self.printStuff("./","sanity_check_u"+app,self.U)

	def sanityCheck(self,app=""):
		self.U.x.array[:]=self.Q.x.array[self.FS_to_FS0]
		self.P.x.array[:]=self.Q.x.array[self.FS_to_FS1]
		#self.Nu.x.array[:]=self.Nu.x.array[:]

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