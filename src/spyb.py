# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
#source /usr/local/bin/dolfinx-real-mode
from mpi4py.MPI import MIN, MAX

from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem import Function, Expression
from dolfinx.mesh import refine, locate_entities
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.geometry import BoundingBoxTree, compute_collisions, compute_colliding_cells

from spy import SPY
from helpers import *

# Swirling Parallel Yaj Baseflow
class SPYB(SPY):
	def __init__(self, params:dict, datapath:str, mesh_name:str, direction_map:dict) -> None:
		super().__init__(params, datapath, mesh_name, direction_map)
		dirCreator(self.baseflow_path)
		
	# Careful here Re is only for printing purposes ; self.Re may be a more involved function
	def baseflow(self,Re:int,S:float,refinement:bool=False,save:bool=True,baseflowInit=None) -> int:
		# Cold initialisation
		if baseflowInit!=None:
			U,_=self.Q.split()
			U.interpolate(baseflowInit)

		# Compute form
		base_form  = self.navierStokes() # No azimuthal decomposition for base flow
		dbase_form = self.linearisedNavierStokes(0) # m=0
		return self.solver(Re,S,base_form,dbase_form,self.Q,save,refinement)

	# Recommand running in real
	def solver(self,Re:int,S:float,base_form:ufl.Form,dbase_form:ufl.Form,q:Function,save:bool=True,refinement:bool=False) -> int:
		# Encapsulations
		problem = NonlinearProblem(base_form,q,self.bcs,dbase_form)
		solver  = NewtonSolver(comm, problem)
		
		# Fine tuning
		solver.convergence_criterion = "incremental"
		solver.relaxation_parameter=self.params['rp'] # Absolutely crucial for convergence
		solver.max_iter=self.params['max_iter']
		solver.rtol=self.params['rtol']
		solver.atol=self.params['atol']
		configureKSP(solver.krylov_solver,self.params)
		if p0: print("Solver launch...",flush=True)
		# Actual heavyweight
		n,_=solver.solve(q)

		if refinement:
			# Locate high error areas
			def high_error(points):
				expr = Expression(self.navierStokesError(),self.TH1.element.interpolation_points())
				res = Function(self.TH)
				res.interpolate(expr)

				bbtree = BoundingBoxTree(self.mesh, 2)
				cells, points_on_proc = [], []
				# Find cells whose bounding-box collide with the the points
				cell_candidates = compute_collisions(bbtree, points)
				# Choose one of the cells that contains the point
				colliding_cells = compute_colliding_cells(self.mesh, cell_candidates, points)
				for i, point in enumerate(points):
					if len(colliding_cells.links(i))>0:
						points_on_proc.append(point)
						cells.append(colliding_cells.links(i)[0])
				# Actual evaluation
				res_at_points = res.eval(points_on_proc, cells)
				# Currently absolutely arbitrary bound of 80% max
				max_res = comm.allreduce(np.max(res_at_points), op=MAX)
				return res_at_points>.8*max_res
			edges = locate_entities(self.mesh, 2, high_error)
			self.mesh.topology.create_entities(1)
			# Mesh refinement
			self.mesh = refine(self.mesh, edges, redistribute=False)
			self.defineFunctionSpaces()
			# Interpolate Newton results on finer mesh
			self.Q.interpolate(q)

		if save:  # Memoisation
			self.saveBaseflow(Re,S)
			U,_=q.split()
			self.printStuff(self.print_path,f"u_Re={Re:d}_S={S:.1f}",U)
		
		return n

	# Helper
	def smoother(self, a, L):
		pb = LinearProblem(a*self.r*ufl.dx, L*self.r*ufl.dx, bcs=self.bcs,
						   petsc_options={"ksp_type":"cg", "pc_type":"gamg", "pc_factor_mat_solver_type":"mumps",
						   				  "ksp_rtol":self.params['rtol'], "ksp_atol":self.params['atol'], "ksp_max_it":self.params['max_iter']})
		if p0: print("Smoothing started...",flush=True)
		u = pb.solve()
		u.x.scatter_forward()
		return u

	# Shorthands
	def smoothenF(self, e:float, F:Function):
		r = self.r
		f, g = ufl.TrialFunction(self.TH1), ufl.TestFunction(self.TH1)
		gd = lambda v: grd(r,self.direction_map['x'],self.direction_map['r'],self.direction_map['th'],v,0)
		F = self.smoother(ufl.inner(f,g)+e*ufl.inner(gd(f),gd(g)),ufl.inner(F,g))
		return F
			
	def smoothenNu(self, e:float):
		self.Nu.x.array[self.Nu.x.array<1e-5]=0 # Somewhat arbitrary cutoff
		self.Nu = self.smoothenF(e,self.Nu)
		self.Nu.x.array[self.Nu.x.array<1e-5]=0 # Important to do it twice
			
	def smoothenP(self, e:float): self.P = self.smoothenF(e,self.P)
	
	def smoothenU(self, e:float, dir=None):
		r = self.r
		u, p, v, s = self.u, self.p, self.v, self.s
		U, P = ufl.split(self.Q)
		gd = lambda v: grd(r,self.direction_map['x'],self.direction_map['r'],self.direction_map['th'],v,0)
		if dir is None: self.Q = self.smoother(ufl.inner(u,v)+e*ufl.inner(gd(u),	 gd(v))		+ufl.inner(p,s),ufl.inner(U,v)+ufl.inner(P,s))
		else: 			self.Q = self.smoother(ufl.inner(u,v)+e*ufl.inner(gd(u[dir]),gd(v[dir]))+ufl.inner(p,s),ufl.inner(U,v)+ufl.inner(P,s))

	def minimumAxial(self) -> float:
		u=self.Q.split()[0].compute_point_values()[:,self.direction_map['x']]
		mu=np.min(u)
		mu=comm.reduce(mu,op=MIN) # minimum across processors
		return mu

	def computeQuiver(self,XYZ:np.array,scale:str) -> list:
		import plotly.graph_objects as go #pip3 install plotly
		
		X,Y,Z = XYZ
		# Evaluation of projected value
		XYZ_p = np.vstack((X,np.sqrt(Y**2+Z**2)))
		U,V,W=self.Q.split()[0].split()
		XYZ_e, U = self.eval(U,XYZ_p.T,XYZ.T)
		_, 	   V = self.eval(V,XYZ_p.T,XYZ.T)
		_, 	   W = self.eval(W,XYZ_p.T,XYZ.T)

		if p0:
			print("Evaluation of baseflow done ! Plotting quiver...",flush=True)
			X,Y,Z = XYZ_e.T
			th = np.arctan2(Z,Y)
			V,W=V*np.cos(th)-W*np.sin(th),W*np.cos(th)+V*np.sin(th)
			return go.Cone(x=X,y=Y,z=Z,u=U.real,v=V.real,w=W.real, # Correcting orientation
						   colorscale=scale,sizemode="scaled",sizeref=1,name="baseflow",opacity=.6,showscale=False)