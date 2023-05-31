# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
from helpers import *

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
		# Eddy viscosity
		self.Nu = Function(self.TH1)

	# Helper
	def loadBaseflow(self,Re:int,S:float,loadNu=True) -> None:
		loadStuff(self.q_path,  {'Re':Re,'S':S},self.Q)
		if loadNu: loadStuff(self.nut_path,{'Re':Re,'S':S},self.Nu)
		
	def saveBaseflow(self,Re:int,S:float,saveNu=False):
		save_str=f"_Re={Re:d}_S={S:.1f}".replace('.',',')
		saveStuff(self.q_path,"q"+save_str,self.Q)
		if saveNu: saveStuff(self.nut_path,"nut"+save_str,self.Nu)
	
	# Heart of this entire code
	def navierStokes(self) -> ufl.Form:
		# Shortforms
		r, v, s = self.r, self.v, self.s
		# Functions
		U, P = ufl.split(self.Q)
		nu = 1/self.Re+self.Nu # Re is taken as in the litterature with respect to D, but the rest is made dimensionless with R
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
		nu = 1/self.Re + self.Nu # Re is taken as in the litterature with respect to D, but the rest is made dimensionless with R
		# More shortforms
		dx,dr,dt=self.direction_map['x'],self.direction_map['r'],self.direction_map['th']
		dv,gd=lambda v,m: div(r,dx,dr,dt,v,m),lambda v,m: grd(r,dx,dr,dt,v,m)
		#cr=lambda v,m: crl(r,dx,dr,dt,self.mesh,v,m)
		# Mass (variational formulation)
		F  = ufl.inner(dv(u,m),   s)
		# Momentum (different test functions and IBP)
		F += ufl.inner(gd(U,0)*u, v) # Convection
		F += ufl.inner(gd(u,m)*U, v)
		F -= ufl.inner(   p,   dv(v,m)) # Pressure
		F += ufl.inner(gd(u,m)+gd(u,m).T,
							   gd(v,m))*nu # Diffusion (grad u.T significant with nut)
		#F -= ufl.inner(2*U[2]*u[2]/r,v[1]) # Cancel out centrifugal force ?
		#F += 2*ufl.inner(ufl.cross(cr(u,m),U)+ufl.cross(cr(U,0),u),v) # Cancel out Coriolis force ?
		#F += 2*ufl.inner(ufl.cross(S*ufl.as_vector([1,0,0]),u),v) # Cancel out Coriolis force ?
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
		U,P=self.Q.split()

		dx,dr=self.direction_map['x'],self.direction_map['r']
		expr=dfx.fem.Expression(U[dx].dx(dx) + (self.r*U[dr]).dx(dr)/self.r,
								self.TH1.element.interpolation_points())
		div = Function(self.TH1)
		div.interpolate(expr)
		self.printStuff("./","sanity_check_div"+app,div)

		FE = ufl.FiniteElement("DG",self.mesh.ufl_cell(),0)
		W = dfx.fem.FunctionSpace(self.mesh,FE)
		p = Function(W)
		p.x.array[:]=comm.rank
		self.printStuff("./","sanity_check_partition"+app,p)
		
		self.printStuff("./","sanity_check_u"+app, U)
		self.printStuff("./","sanity_check_p"+app, P)
		# nut may not be a Function
		try: self.printStuff("./","sanity_check_nut"+app,self.Nu)
		except TypeError: pass

	def sanityCheckBCs(self,str=""):
		v=Function(self.TH)
		v.vector.zeroEntries()
		v.x.array[self.dofs]=np.ones(self.dofs.size)
		v,_=v.split()
		self.printStuff("./","sanity_check_bcs"+str,v)