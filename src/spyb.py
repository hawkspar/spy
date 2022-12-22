# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import shutil
import numpy as np
from copy import deepcopy
from spy import SPY, dirCreator
from petsc4py import PETSc as pet
from dolfinx.fem import Function
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.geometry import BoundingBoxTree, compute_collisions, compute_colliding_cells
from dolfinx.mesh import locate_entities, meshtags, refine
from mpi4py.MPI import COMM_WORLD as comm, MIN

p0=comm.rank==0

# Swirling Parallel Yaj Baseflow
class SPYB(SPY):
	def __init__(self, params:dict, datapath:str, direction_map:dict) -> None:
		super().__init__(params, datapath, direction_map)
		dirCreator(self.baseflow_path)
	
	def smoothenBaseflow(self,bcs_u,weak_bcs_u):
		self.Q.x.array[self.FS_to_FS0]=self.smoothen(5e-3,self.U,self.FS0,bcs_u,weak_bcs_u)
		self.Q.x.array[self.FS_to_FS1]=self.smoothen(5e-2,self.P,self.FS1,[],lambda spy,p,s:0)
		self.Q.x.scatter_forward()

	# Careful here Re is only for printing purposes ; self.Re is a more involved function
	def baseflow(self,Re:int,nut:int,S:float,dist,weak_bcs=lambda spy,u,p,m=0: 0,save:bool=True,baseflowInit=None,stabilise=False):
		# Cold initialisation
		if baseflowInit!=None:
			U,_,_=self.Q.split()
			U.interpolate(baseflowInit)

		# Compute form
		base_form  = self.navierStokes(weak_bcs,dist,stabilise) # No azimuthal decomposition for base flow
		#dbase_form = self.linearisedNavierStokes(weak_bcs,0,dist,stabilise) # m=0
		
		# Encapsulations
		problem = NonlinearProblem(base_form,self.Q,bcs=self.bcs)#,J=dbase_form)
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
		solver.solve(self.Q)

		if save:  # Memoisation
			if S==0: app=f"_S={S:d}_Re={Re:d}_nut={nut:d}"
			else: 	 app=f"_S={S:00.3f}_Re={Re:d}_nut={nut:d}"
			self.saveBaseflow(app)
			U,_,_=self.Q.split()
			self.printStuff(self.print_path,"u"+app,U)

		if refine:
			# Save Newton results and residual
			Q=deepcopy(self.Q)
			res=Function(self.FS)
			res.vector=solver.b
			bbtree = BoundingBoxTree(self.mesh, 2)
			def high_error(x):
				# Find cells whose bounding-box collide with the the points
				cell_candidates = compute_collisions(bbtree, x)
				# Choose one of the cells that contains the point
				cells = compute_colliding_cells(self.mesh, cell_candidates, x)
				res = res.eval(x, cells)
				return res>.8*np.max(res)
			facets = locate_entities(self.mesh, 2, high_error)
			refine_indices.append(facets)
			refine_markers.append(np.full(len(facets), 1))
			# Only markers should be int8
			refine_indices = np.array(np.hstack(refine_indices), dtype=np.int32)
			refine_markers = np.array(np.hstack(refine_markers), dtype=np.int8)
			# Indices sent into meshtags should be sorted
			sort = np.argsort(refine_indices)
			refine_tag = meshtags(self.mesh, 2, refine_indices[sort], refine_markers[sort])

			self.mesh.topology.create_entities(1)
			self.mesh = refine(self.mesh, refine_tag)
			self.defineFunctionSpaces()
			self.Q.interpolate(Q)

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