# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import shutil, ufl
import numpy as np
from copy import copy
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
	def __init__(self, params:dict, datapath:str, mesh_name:str, direction_map:dict) -> None:
		super().__init__(params, datapath, mesh_name, direction_map)
		dirCreator(self.baseflow_path)
	
	def smoothenBaseflow(self,bcs_u,weak_bcs_u):
		self.Q.x.array[self.FS_to_FS0]=self.smoothen(5e-3,self.U,self.FS0,bcs_u,weak_bcs_u)
		self.Q.x.array[self.FS_to_FS1]=self.smoothen(5e-2,self.P,self.FS1,[],lambda spy,p,s:0)
		self.Q.x.scatter_forward()

	# Careful here Re is only for printing purposes ; self.Re is a more involved function
	def baseflow(self,Re:int,S:float,dist,weak_bcs=lambda spy,u,p,m=0: 0,save:bool=True,refinement:bool=False,baseflowInit=None,stabilise=False):
		# Cold initialisation
		if baseflowInit!=None:
			U,_,_=self.Q.split()
			U.interpolate(baseflowInit)

		# Compute form
		base_form  = self.navierStokes(weak_bcs,dist,stabilise) # No azimuthal decomposition for base flow
		dbase_form = self.linearisedNavierStokes(weak_bcs,0,dist,stabilise) # m=0
		
		# Encapsulations
		problem = NonlinearProblem(base_form,self.Q,bcs=self.bcs,J=dbase_form)
		solver  = NewtonSolver(comm, problem)
		
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
		n,converged=solver.solve(self.Q)

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
			Q=copy(self.Q)
			self.Q.interpolate(Q)

		if save:  # Memoisation
			if S==0: app=f"_S={S:d}_Re={Re:d}"
			else: 	 app=f"_S={S:00.3f}_Re={Re:d}"
			self.saveBaseflow(app)
			U,_,_=self.Q.split()
			self.printStuff(self.print_path,"u"+app,U)
		
		return n

	# To be run in real mode
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