# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import os, ufl
import warnings
import numpy as np
from spy import spy
import dolfinx as dfx
from dolfinx.log import set_log_level, LogLevel
from dolfinx.io import XDMFFile
from petsc4py import PETSc as pet
from mpi4py.MPI import COMM_WORLD, MIN

p0=COMM_WORLD.rank==0

# Swirling Parallel Yaj Baseflow
class spyb(spy):
	def __init__(self, meshpath: str, datapath: str, dM:float) -> None:
		super().__init__(meshpath, datapath, dM)
		self.BoundaryConditions(0)
		
	# Memoisation routine - find closest in S
	def HotStart(self,Re, S) -> None:
		closest_file_name=self.datapath+"last_baseflow_real.dat"
		file_names = [f for f in os.listdir(self.datapath+self.baseflow_path+'dat_real/') if f[-3:]=="dat"]
		d=np.infty
		for file_name in file_names:
			Ref = float(file_name[12:16]) # Take advantage of file format
			Sf = float(file_name[19:24]) # Take advantage of file format
			fd = abs(S-Sf)+abs(Re-Ref)
			if fd<d: d,closest_file_name=fd,self.datapath+self.baseflow_path+'dat_real/'+file_name
		viewer = pet.Viewer().createMPIIO(closest_file_name, 'r', COMM_WORLD)
		self.q.vector.load(viewer)
		self.q.vector.ghostUpdate(addv=pet.InsertMode.INSERT, mode=pet.ScatterMode.FORWARD)
		print("Loaded "+closest_file_name+" as part of memoisation scheme")

	# Baseflow (really only need DirichletBC objects)
	def BoundaryConditions(self,S:float) -> None:
		# ----------------------------------------------------------------
		# Inlet azimuthal velocity
		# ----------------------------------------------------------------
		
		# Relevant spaces
		sub_space_th=self.Space.sub(0).sub(2)
		sub_space_th_collapsed=sub_space_th.collapse()
		class InletAzimuthalVelocity():
			def __init__(self, S, r_max): self.S, self.r_max = S, r_max
			def __call__(self, x): 
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					return self.S*(x[1]*(x[1]<1)+(1/x[1]-1/self.r_max)/(1-1/self.r_max)*(x[1]>1)).astype(pet.ScalarType)

		# Create a adjustable dolfinx Function
		self.u_inlet_th=dfx.Function(sub_space_th_collapsed)
		self.inlet_azimuthal_velocity=InletAzimuthalVelocity(S,self.r_max) # Required to smoothly increase S
		self.u_inlet_th.interpolate(self.inlet_azimuthal_velocity)
		self.u_inlet_th.x.scatter_forward()
		
		# Degrees of freedom & BC
		dofs_inlet_th = dfx.fem.locate_dofs_geometrical((sub_space_th, sub_space_th_collapsed), self.inlet)
		bcs_inlet_th = dfx.DirichletBC(self.u_inlet_th, dofs_inlet_th, sub_space_th) # u_th=S*r at x=0 (inside nozzle, e^-r outside)
		
		# ----------------------------------------------------------------
		# Nozzle azimuthal velocity
		# ----------------------------------------------------------------

		class NozzleAzimuthalVelocity():
			def __init__(self, S): self.S = S
			def __call__(self, x): return self.S*np.ones(x.shape[1]).astype(pet.ScalarType) # Simple constant function

		# Create a adjustable dolfinx Function
		self.u_nozzle_th=dfx.Function(sub_space_th_collapsed)
		self.nozzle_azimuthal_velocity=NozzleAzimuthalVelocity(S) # Required to smoothly increase S
		self.u_nozzle_th.interpolate(self.nozzle_azimuthal_velocity)
		self.u_nozzle_th.x.scatter_forward()
		
		# Degrees of freedom & BC
		dofs_nozzle_th = dfx.fem.locate_dofs_geometrical((sub_space_th, sub_space_th_collapsed), self.nozzle)
		bcs_nozzle_th = dfx.DirichletBC(self.u_nozzle_th, dofs_nozzle_th, sub_space_th) # u_th=S at nozzle
		
		# ----------------------------------------------------------------
		# Inlet axial velocity
		# ----------------------------------------------------------------

		# Compute spaces
		sub_space_x=self.Space.sub(0).sub(0)
		sub_space_x_collapsed=sub_space_x.collapse()

		# Simple tanh flow in nozzle only
		self.u_inlet_x=dfx.Function(sub_space_x_collapsed)
		self.u_inlet_x.interpolate(lambda x: (x[1]<1)*np.tanh(5*(1-x[1]))) # For now no co-flow
		self.u_inlet_x.x.scatter_forward()
		
		# Degrees of freedom
		dofs_inlet_x = dfx.fem.locate_dofs_geometrical((sub_space_x, sub_space_x_collapsed), self.inlet)
		bcs_inlet_x = dfx.DirichletBC(self.u_inlet_x, dofs_inlet_x, sub_space_x) # u_x=th(r) at x=0 (inside nozzle, 0 outside)

		# Actual BCs
		self.bcs = [bcs_inlet_x, bcs_inlet_th, bcs_nozzle_th] # x=X entirely handled by implicit Neumann
		
		# Handle homogeneous boundary conditions
		homogeneous_boundaries=[(self.inlet,['r']),(self.top,['r','th']),(self.symmetry,['r','th']),(self.nozzle,['x','r'])]
		for tup in homogeneous_boundaries:
			marker,directions=tup
			for direction in directions:
				_, bcs=self.ConstantBC(direction,marker)
				self.bcs.append(bcs)
	
	def Baseflow(self,hot_start:bool,save:bool,Re:float,S:float):
		# Apply new BC
		self.inlet_azimuthal_velocity.S=S
		self.u_inlet_th.interpolate(self.inlet_azimuthal_velocity)
		self.u_inlet_th.x.scatter_forward()
		self.nozzle_azimuthal_velocity.S=S
		self.u_nozzle_th.interpolate(self.nozzle_azimuthal_velocity)
		self.u_nozzle_th.x.scatter_forward()
		# Memoisation
		if hot_start: self.HotStart(Re,S)
		# Compute form
		base_form  = self.NavierStokes(Re) #no azimuthal decomposition for base flow
		dbase_form = self.LinearisedNavierStokes(Re,0) # m=0

		# Encapsulations
		problem = dfx.fem.NonlinearProblem(base_form,self.q,bcs=self.bcs,J=dbase_form)
		solver  = dfx.NewtonSolver(COMM_WORLD, problem)
		# Fine tuning
		solver.convergence_criterion = "incremental"
		solver.relaxation_parameter=self.rp # Absolutely crucial for convergence
		solver.max_iter=self.max_iter
		solver.rtol=self.rtol
		solver.atol=self.atol
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
			with XDMFFile(COMM_WORLD, self.datapath+self.baseflow_path+"print/u_Re="+f"{Re:04.0f}"+"_S="+f"{S:00.3f}"+".xdmf", "w") as xdmf:
				xdmf.write_mesh(self.mesh)
				xdmf.write_function(u)
			viewer = pet.Viewer().createMPIIO(self.datapath+self.baseflow_path+"dat_real/baseflow_Re="+f"{Re:04.0f}"+"_S="+f"{S:00.3f}"+".dat", 'w', COMM_WORLD)
			self.q.vector.view(viewer)
			if COMM_WORLD.rank==0: print(".xmdf, .dat written!")

	# To be run in real mode
	def BaseflowRange(self,Res,Ss) -> None:
		for Re in Res:		   # Loop on Reynolds number first
			for S in Ss: 	   # Then on swirl intensity
				if COMM_WORLD.rank==0:
					print("##########################")
					print("Reynolds number: ",	    Re)
					print("Swirl intensity: ",	    S)
				self.Baseflow(True,True,Re,S)
				#self.Baseflow(Re>Res[0] or S>Ss[0],True,Re,S)
				
		#write result of current mu
		u,p=self.q.split()
		with XDMFFile(COMM_WORLD, self.datapath+"last_u.xdmf", "w") as xdmf:
			xdmf.write_mesh(self.mesh)
			xdmf.write_function(u)
		viewer = pet.Viewer().createMPIIO(self.datapath+"last_baseflow_real.dat", 'w', COMM_WORLD)
		self.q.vector.view(viewer)
		print("Last checkpoint written!")

	def MinimumAxial(self) -> float:
		u,p=self.q.split()
		u=u.compute_point_values()[:,self.direction_map['x']]
		mu=np.min(u)
		mu=COMM_WORLD.reduce(mu,op=MIN) # minimum across processors
		return mu