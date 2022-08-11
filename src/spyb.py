# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import shutil
import numpy as np
from spy import SPY, dirCreator
from petsc4py import PETSc as pet
from dolfinx.fem import NonlinearProblem
from dolfinx import Function, NewtonSolver
from mpi4py.MPI import COMM_WORLD as comm, MIN

pet.Options().setValue('-snes_linesearch_type', 'basic') # classical Newton method

p0=comm.rank==0

# Swirling Parallel Yaj Baseflow
class SPYB(SPY):
	def __init__(self, params:dict, datapath:str, Ref, nutf, direction_map:dict, InletAzimuthalVelocity) -> None:
		super().__init__(params, datapath, Ref, nutf, direction_map)
		sub_space_th=self.TH.sub(0).sub(direction_map['th'])
		sub_space_th_collapsed=sub_space_th.collapse()

		# Modified vortex that goes to zero at top boundary
		self.u_inlet_th=Function(sub_space_th_collapsed)
		self.inlet_azimuthal_velocity=InletAzimuthalVelocity(0)
		dirCreator(self.baseflow_path)
	
	# Careful here Re is only for printing purposes ; self.Re is a more involved function
	def baseflow(self,Re:int,S:float,hot_start:bool,save:bool=True,baseflowInit=None):
		# Apply new BC
		self.inlet_azimuthal_velocity.S=S
		self.u_inlet_th.interpolate(self.inlet_azimuthal_velocity)
		# Cold initialisation
		if baseflowInit!=None:
			u,p=self.q.split()
			u.interpolate(baseflowInit)
		# Memoisation
		elif hot_start:
			self.loadStuff([S,Re],self.dat_real_path,['S','Re'],self.q.vector)
		
		# Load turbulent viscosity if need be
		self.nutf(S)

		# Compute form
		base_form  = self.navierStokes() #no azimuthal decomposition for base flow
		dbase_form = self.linearisedNavierStokes(0) # m=0
		
		# Encapsulations
		problem = NonlinearProblem(base_form,self.q,bcs=self.bcs,J=dbase_form)
		solver  = NewtonSolver(comm, problem)
		
		# Fine tuning
		solver.convergence_criterion = "incremental"
		solver.report = True
		#log.set_log_level(log.LogLevel.INFO)
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
		if p0: print("Solver launch...")
		# Actual heavyweight
		solver.solve(self.q)

		if save:  # Memoisation
			u,p = self.q.split()
			self.saveStuff(self.print_path,f"u_S={S:00.3f}_Re={Re:d}",u)
			self.saveStuffMPI(self.dat_real_path,f"baseflow_S={S:00.3f}_Re={Re:d}",self.q.vector)
			if p0: print(".xmdf, .dat written!")

	# To be run in real mode
	# DESTRUCTIVE !
	def baseflowRange(self,Ss) -> None:
		shutil.rmtree(self.dat_real_path)
		shutil.rmtree(self.print_path)
		for S in Ss: 	   # Then on swirl intensity
			if p0:
				print('#'*25)
				print("Swirl intensity: ", S)
			self.baseflow(S!=Ss[0],True,S)

		if p0: print("Last checkpoint written!")

	def minimumAxial(self) -> float:
		u,p=self.q.split()
		u=u.compute_point_values()[:,self.direction_map['x']]
		mu=np.min(u)
		mu=comm.reduce(mu,op=MIN) # minimum across processors
		return mu