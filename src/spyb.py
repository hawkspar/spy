# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import shutil, ufl
import numpy as np
from spy import SPY, dirCreator
from petsc4py import PETSc as pet
from mpi4py.MPI import COMM_WORLD as comm, MIN, MAX
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem import Function, Expression
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import locate_entities, refine
from dolfinx.geometry import BoundingBoxTree, compute_collisions, compute_colliding_cells

p0=comm.rank==0

# Swirling Parallel Yaj Baseflow
class SPYB(SPY):
	def __init__(self, params:dict, datapath:str, Ref, nutf, direction_map:dict, InletAzimuthalVelocity, C:bool=False, full_TH:bool=True) -> None:
		super().__init__(params, datapath, Ref, nutf, direction_map, C)
		if full_TH: th_space,_=self.u_space.sub(direction_map['th']).collapse()
		else:
			FE_vector = ufl.VectorElement("CG",self.mesh.ufl_cell(),2,3)
			V = FunctionSpace(self.mesh,FE_vector)
			th_space,_=V.sub(direction_map['th']).collapse()

		# Modified vortex that goes to zero at top boundary
		self.u_inlet_th=Function(th_space)
		self.inlet_azimuthal_velocity=InletAzimuthalVelocity(0)
		dirCreator(self.baseflow_path)
	
	def smoothenBaseflow(self,bcs_u,weak_bcs_u):
		self.Q.x.array[self.FS_to_FS0]=self.smoothen(5e-3,self.U,self.FS0,bcs_u,weak_bcs_u)
		self.Q.x.array[self.FS_to_FS1]=self.smoothen(5e-2,self.P,self.FS1,[],lambda spy,p,s:0)
		self.Q.x.scatter_forward()

	def smoothen(self,e):
		FE_scalar = ufl.FiniteElement("CG",self.mesh.ufl_cell(),1)
		V = FunctionSpace(self.mesh,FE_scalar)
		u,v=ufl.TrialFunction(V),ufl.TestFunction(V)
		P,r=self.P,self.r
		grd,div=lambda v: self.grd_nor(v,0),lambda v: self.div_nor(v,0)
		a=ufl.inner(u,v)
		a+=e*ufl.inner(grd(u),grd(v))
		L=ufl.inner(P,v)
		pb = dfx.fem.petsc.LinearProblem(a*r*ufl.dx, L*r*ufl.dx, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
		self.P = pb.solve()
		if p0: print("Smoothing finished !",flush=True)
	
	# Careful here Re is only for printing purposes ; self.Re is a more involved function
	def baseflow(self,Re:int,S:float,hot_start:bool,save:bool=True,baseflowInit=None):
		# Apply new BC
		self.inlet_azimuthal_velocity.S=S
		self.u_inlet_th.interpolate(self.inlet_azimuthal_velocity)
		# Cold initialisation
		if baseflowInit!=None:
			U,_=self.Q.split()
			U.interpolate(baseflowInit)

		self.Q = Function(self.TH)
		_, V_to_TH = self.TH.sub(0).collapse()
		_, W_to_TH = self.TH.sub(1).collapse()
		self.Q.x.array[V_to_TH] = self.U.x.array
		self.Q.x.array[W_to_TH] = self.P.x.array

		# Compute form
		base_form  = self.navierStokes() #no azimuthal decomposition for base flow
		#dbase_form = self.linearisedNavierStokes(0) # m=0
		
		return Re,n
		
	# Recomand running in real
	def solver(self,Re:int,nut:int,S:float,base_form:ufl.Form,dbase_form:ufl.Form,q:Function,save:bool=True,refinement:bool=False) -> int:
		# Encapsulations
		problem = NonlinearProblem(base_form,self.Q,bcs=self.bcs)#,J=dbase_form)
		solver  = NewtonSolver(comm, problem)

		if p0: print("self.U.size=",self.U.x.array.size,flush=True)
		
		# Fine tuning
		solver.convergence_criterion = "incremental"
		solver.relaxation_parameter=self.params['rp'] # Absolutely crucial for convergence
		solver.max_iter=self.params['max_iter']
		solver.rtol=self.params['rtol']
		solver.atol=self.params['atol']
		ksp = solver.krylov_solver
		opts = pet.Options()
		option_prefix = ksp.getOptionsPrefix()
		opts[f"{option_prefix}ksp_type"] = "preonly"
		opts[f"{option_prefix}pc_type"] = "lu"
		opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
		ksp.setFromOptions()
		if p0: print("Solver launch...",flush=True)
		# Actual heavyweight
		n,converged=solver.solve(q)

		if refinement:
			# Locate high error areas
			def high_error(points):
				expr = Expression(self.navierStokesError(),self.FS1.element.interpolation_points())
				res = Function(self.FS)
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
			self.printStuff(self.print_path,f"u_S={S:00.3f}_Re={Re:d}",self.U)
			self.saveBaseflow("_S={S:00.3f}_Re={Re:d}")
			if p0: print(".xmdf, .dat written!",flush=True)

	# DESTRUCTIVE !
	def baseflowRange(self,Ss) -> None:
		shutil.rmtree(self.u_path)
		shutil.rmtree(self.p_path)
		shutil.rmtree(self.print_path)
		for S in Ss: 	   # Then on swirl intensity
			if p0:
				print('#'*25)
				print("Swirl intensity: ", S)
			self.baseflow(S!=Ss[0],True,S)

		if p0: print("Last checkpoint written!",flush=True)

	def minimumAxial(self) -> float:
		u=self.U.compute_point_values()[:,self.direction_map['x']]
		mu=np.min(u)
		mu=comm.reduce(mu,op=MIN) # minimum across processors
		return mu