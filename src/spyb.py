# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import shutil, ufl
import numpy as np
from petsc4py import PETSc as pet
from spy import SPY, dirCreator, grd
from mpi4py.MPI import COMM_WORLD as comm, MIN, MAX

from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem import Function, Expression
from dolfinx.mesh import locate_entities, refine
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.geometry import BoundingBoxTree, compute_collisions, compute_colliding_cells

p0=comm.rank==0

# Swirling Parallel Yaj Baseflow
class SPYB(SPY):
	def __init__(self, params:dict, datapath:str, mesh_name:str, direction_map:dict) -> None:
		super().__init__(params, datapath, mesh_name, direction_map)
		dirCreator(self.baseflow_path)

	def smoothen(self, e:float):
		r=self.r
		u, p  = ufl.split(self.trial)
		U, P  = ufl.split(self.Q)
		v, s  = ufl.split(self.test)
		gd=lambda v: grd(r,self.direction_map['x'],self.direction_map['r'],self.direction_map['th'],v,0)
		a=ufl.inner(u,v)+e*ufl.inner(gd(u),gd(v))+ufl.inner(p,s)
		L=ufl.inner(U,v)+ufl.inner(P,s)
		pb = LinearProblem(a*r*ufl.dx, L*r*ufl.dx, bcs=self.bcs, petsc_options={"ksp_type": "cg", "ksp_rtol":1e-6, "ksp_atol":1e-9, "ksp_max_it": 100, "pc_type": "gamg", "pc_factor_mat_solver_type": "mumps"})
		if p0: print("Smoothing started...",flush=True)
		self.Q=pb.solve()

	# Careful here Re is only for printing purposes ; self.Re may be a more involved function
	def baseflow(self,Re:int,S:float,dist,weak_bcs:tuple=(0,0),refinement:bool=False,save:bool=True,baseflowInit=None) -> int:
		# Cold initialisation
		if baseflowInit!=None:
			U,_=self.Q.split()
			U.interpolate(baseflowInit)

		# Compute form
		base_form  = self.navierStokes(self.Q,self.test,dist)#+weak_bcs[0] # No azimuthal decomposition for base flow
		dbase_form = self.linearisedNavierStokes(self.trial,self.Q,self.test,0,dist)#+weak_bcs[1] # m=0
		return self.solver(Re,S,base_form,dbase_form,self.Q,refinement=refinement,save=save)

	def corrector(self,Q0,dQ,dRe,h,Re0:int,S:float,d,weak_bcs:tuple) -> int:
		Qe=self.extend(self.Q)
		Q0=self.extend(Q0)
		dQ=self.extend(dQ)
		u, p, nu, re= ufl.split(self.triale)
		U, P, Nu, Re= ufl.split(Qe)
		_, _, _,  w = ufl.split(self.teste)
		U0,P0,Nu0,_ = ufl.split(Q0)
		dU,dP,dNu,_ = ufl.split(dQ)
		self.Re=Re
		base_form  = self.navierStokes(Qe,self.teste,d,extended=True)+weak_bcs[0]+\
					 ufl.inner(ufl.inner(dU,U-U0)+ufl.inner(dP,P-P0)+ufl.inner(dNu,Nu-Nu0)+dRe*(Re-Re0)-h,w) # No azimuthal decomposition for base flow
		dbase_form = self.linearisedNavierStokes(self.triale,Qe,self.teste,0,d,extended=True)+weak_bcs[1]+\
					 ufl.inner(ufl.inner(dU,u)+ufl.inner(dP,p)+ufl.inner(dNu,nu)+dRe*re,w) # No azimuthal decomposition for base flow
		n=self.solver(Re0,S,base_form,dbase_form,Qe,save=False)
		self.Q,Re=self.revert(Qe)
		
		return Re,n
		
	# Recomand running in real
	def solver(self,Re:int,S:float,base_form:ufl.Form,dbase_form:ufl.Form,q:Function,save:bool=True,refinement:bool=False) -> int:
		# Encapsulations
		problem = NonlinearProblem(base_form,q,bcs=self.bcs,J=dbase_form)
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
		n,converged=solver.solve(q)

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
			self.printStuff(self.print_path,"u"+app,U)
		
		return n

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