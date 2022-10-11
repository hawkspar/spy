# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import shutil, ufl
import numpy as np
from spy import SPY, dirCreator
from petsc4py import PETSc as pet
from dolfinx.fem import FunctionSpace
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py.MPI import COMM_WORLD as comm, MIN

#pet.Options().setValue('-snes_linesearch_type', 'basic') # classical Newton method

p0=comm.rank==0

# Swirling Parallel Yaj Baseflow
class SPYB(SPY):
	def __init__(self, params:dict, datapath:str, Ref, nutf, direction_map:dict, C:bool=False) -> None:
		super().__init__(params, datapath, Ref, nutf, direction_map, C)
		dirCreator(self.baseflow_path)

	def smoothenBaseflow(self):
		FE_vector=ufl.VectorElement("CG",self.mesh.ufl_cell(),2,3)
		FE_scalar=ufl.FiniteElement("CG",self.mesh.ufl_cell(),1)
		U = FunctionSpace(self.mesh,FE_vector)
		V = FunctionSpace(self.mesh,FE_scalar)
		self.Q.x.array[self.TH0_to_TH]=self.smoothen(1e-2,self.U,U).x.array
		self.Q.x.array[self.TH1_to_TH]=self.smoothen(1e-2,self.P,V).x.array

	# Careful here Re is only for printing purposes ; self.Re is a more involved function
	def baseflow(self,Re:int,S:float,hot_start:bool,save:bool=True,baseflowInit=None):
		# Apply new BC
		self.inlet_azimuthal_velocity.S=S
		# Cold initialisation
		if baseflowInit!=None:
			U,P=self.Q.split()
			U.interpolate(baseflowInit)
		# Memoisation
		elif hot_start:	self.loadBaseflow(S,Re)

		# Compute form
		base_form  = self.navierStokes() #no azimuthal decomposition for base flow
		dbase_form = self.linearisedNavierStokes(0) # m=0
		
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
		opts[f"{option_prefix}pc_type"] = "lu"
		opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
		ksp.setFromOptions()
		if p0: print("Solver launch...",flush=True)
		# Actual heavyweight
		solver.solve(self.Q)

		if save:  # Memoisation
			self.saveBaseflow(f"_S={S:00.3f}_Re={Re:d}")
			U,P=self.Q.split()
			self.printStuff(self.print_path,f"u_S={S:00.3f}_Re={Re:d}",U)

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