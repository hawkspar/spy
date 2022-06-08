# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import os, shutil
import numpy as np
from spy import SPY
import dolfinx as dfx
from dolfinx.io import XDMFFile
from petsc4py import PETSc as pet
from mpi4py.MPI import COMM_WORLD as comm, MIN

p0=comm.rank==0

# Swirling Parallel Yaj Baseflow
class SPYB(SPY):
	def __init__(self, params:dict, datapath:str, Ref, nutf, direction_map:dict, InletAzimuthalVelocity) -> None:
		super().__init__(params, datapath, Ref, nutf, direction_map)
		sub_space_th=self.TH.sub(0).sub(direction_map['th'])
		sub_space_th_collapsed=sub_space_th.collapse()

		# Modified vortex that goes to zero at top boundary
		self.u_inlet_th=dfx.Function(sub_space_th_collapsed)
		self.inlet_azimuthal_velocity=InletAzimuthalVelocity(0)
		
	# Memoisation routine - find closest in S
	def hotStart(self, S) -> None:
		self.loadStuff(S,"last_baseflow.dat",self.dat_real_path,11,self.q.vector)
	
	def baseflow(self,hot_start:bool,save:bool,S:float):
		# Apply new BC
		self.inlet_azimuthal_velocity.S=S
		self.u_inlet_th.interpolate(self.inlet_azimuthal_velocity)
		
		# Memoisation
		if hot_start: self.hotStart(S)
		# Compute form
		base_form  = self.navierStokes() #no azimuthal decomposition for base flow
		dbase_form = self.linearisedNavierStokes(0) # m=0

		# Encapsulations
		problem = dfx.fem.NonlinearProblem(base_form,self.q,bcs=self.bcs,J=dbase_form)
		solver  = dfx.NewtonSolver(comm, problem)
		# Fine tuning
		solver.convergence_criterion = "incremental"
		solver.relaxation_parameter=self.params['rp'] # Absolutely crucial for convergence
		solver.max_iter=self.params['max_iter']
		solver.rtol=self.params['rtol']
		solver.atol=self.params['atol']
		solver.report = True
		ksp = solver.krylov_solver
		opts = pet.Options()
		option_prefix = ksp.getOptionsPrefix()
		opts[f"{option_prefix}pc_type"] = "lu"
		opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
		#set_log_level(LogLevel.INFO)
		ksp.setFromOptions()
		# Actual heavyweight
		try: solver.solve(self.q)
		except RuntimeError: pass
		self.q.x.scatter_forward()

		if save:  # Memoisation
			u,p = self.q.split()
			if not os.path.isdir(self.dat_real_path): os.mkdir(self.dat_real_path)
			with XDMFFile(comm, self.print_path+f"u_S={S:00.3f}.xdmf", "w") as xdmf:
				xdmf.write_mesh(self.mesh)
				xdmf.write_function(u)
			if not os.path.isdir(self.print_path): os.mkdir(self.print_path)
			viewer = pet.Viewer().createMPIIO(self.dat_real_path+f"baseflow_S={S:00.3f}.dat", 'w', comm)
			self.q.vector.view(viewer)
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
		
		#write result of current mu
		u,p=self.q.split()
		with XDMFFile(comm, self.print_path+"last_u.xdmf", "w") as xdmf:
			xdmf.write_mesh(self.mesh)
			xdmf.write_function(u)
		viewer = pet.Viewer().createMPIIO(self.dat_real_path+"last_baseflow.dat", 'w', comm)
		self.q.vector.view(viewer)
		if p0: print("Last checkpoint written!")

	def minimumAxial(self) -> float:
		u,p=self.q.split()
		u=u.compute_point_values()[:,self.direction_map['x']]
		mu=np.min(u)
		mu=comm.reduce(mu,op=MIN) # minimum across processors
		return mu