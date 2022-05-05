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
from mpi4py.MPI import COMM_WORLD, MIN

p0=COMM_WORLD.rank==0

# Swirling Parallel Yaj Baseflow
class SPYB(SPY):
	def __init__(self, params:dict, datapath:str, Ref, nutf,
				 boundaryConditions) -> None:
		super().__init__(params, datapath, Ref, nutf)
		self.boundaryConditions=lambda S: boundaryConditions(self,S)
		self.boundaryConditions(0)
		
	# Memoisation routine - find closest in S
	def hotStart(self, S) -> None:
		closest_file_name=self.case_path+"last_baseflow_real.dat"
		if not os.path.isdir(self.dat_real_path): os.mkdir(self.dat_real_path)
		file_names = [f for f in os.listdir(self.dat_real_path) if f[-3:]=="dat"]
		d=np.infty
		for file_name in file_names:
			Sf = float(file_name[11:16])
			fd = abs(S-Sf)#+abs(Re-Ref)
			if fd<d: d,closest_file_name=fd,self.dat_real_path+file_name
		viewer = pet.Viewer().createMPIIO(closest_file_name, 'r', COMM_WORLD)
		self.q.vector.load(viewer)
		self.q.vector.ghostUpdate(addv=pet.InsertMode.INSERT, mode=pet.ScatterMode.FORWARD)
		if COMM_WORLD.rank==0:
			print("Loaded "+closest_file_name+" as part of memoisation scheme")
	
	def baseflow(self,hot_start:bool,save:bool,S:float):
		# Apply new BC
		self.inlet_azimuthal_velocity.S=S
		self.u_inlet_th.interpolate(self.inlet_azimuthal_velocity)
		self.u_inlet_th.x.scatter_forward()
		
		# Memoisation
		if hot_start: self.hotStart(S)
		# Compute form
		base_form  = self.navierStokes() #no azimuthal decomposition for base flow
		dbase_form = self.linearisedNavierStokes(0) # m=0

		# Encapsulations
		problem = dfx.fem.NonlinearProblem(base_form,self.q,bcs=self.bcs,J=dbase_form)
		solver  = dfx.NewtonSolver(COMM_WORLD, problem)
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
			with XDMFFile(COMM_WORLD, self.print_path+"u_S="+f"{S:00.3f}"+".xdmf", "w") as xdmf:
				xdmf.write_mesh(self.mesh)
				xdmf.write_function(u)
			if not os.path.isdir(self.print_path): os.mkdir(self.print_path)
			viewer = pet.Viewer().createMPIIO(self.dat_real_path+"baseflow_S="+f"{S:00.3f}"+".dat", 'w', COMM_WORLD)
			self.q.vector.view(viewer)
			if COMM_WORLD.rank==0: print(".xmdf, .dat written!")

	# To be run in real mode
	# DESTRUCTIVE !
	def baseflowRange(self,Ss) -> None:
		shutil.rmtree(self.dat_real_path)
		shutil.rmtree(self.print_path)
		for S in Ss: 	   # Then on swirl intensity
			if COMM_WORLD.rank==0:
				print("##########################")
				print("Swirl intensity: ",	    S)
			self.baseflow(True,True,S)
				
		#write result of current mu
		u,p=self.q.split()
		with XDMFFile(COMM_WORLD, self.case_path+"last_u.xdmf", "w") as xdmf:
			xdmf.write_mesh(self.mesh)
			xdmf.write_function(u)
		viewer = pet.Viewer().createMPIIO(self.case_path+"last_baseflow_real.dat", 'w', COMM_WORLD)
		self.q.vector.view(viewer)
		if COMM_WORLD.rank==0: print("Last checkpoint written!")

	def minimumAxial(self) -> float:
		u,p=self.q.split()
		u=u.compute_point_values()[:,self.direction_map['x']]
		mu=np.min(u)
		mu=COMM_WORLD.reduce(mu,op=MIN) # minimum across processors
		return mu