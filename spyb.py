# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import os, shutil
import numpy as np
from spy import spy
import dolfinx as dfx
from dolfinx.io import XDMFFile
from petsc4py import PETSc as pet
from mpi4py.MPI import COMM_WORLD, MIN

p0=COMM_WORLD.rank==0

# Swirling Parallel Yaj Baseflow
class spyb(spy):
	def __init__(self,datapath:str,Re:float,meshpath:str="") -> None:
		super().__init__(datapath,Re,meshpath)
		self.BoundaryConditions(0)
		
	# Memoisation routine - find closest in S
	def HotStart(self, S) -> None:
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

	# Baseflow (really only need DirichletBC objects) enforces :
	# u_x=1, u_r=0 & u_th=gb at inlet (velocity control)
	# u_r=0, u_th=0 for symmetry axis (derived from mass csv as r->0)
	# Nothing at outlet
	# u_r=0, u_th=0 at top (Meliga paper, no slip)
	# However there are hidden BCs in the weak form !
	# Because of the IPP, we have nu grad u.n=p n everywhere. This gives :
	# d_ru_x=0 for symmetry axis (momentum csv r as r->0)
	# d_xu_x/Re=p, d_xu_r=0, d_xu_th=0 at outlet (free flow)
	# d_ru_x=0 at top (Meliga paper, no slip)
	def BoundaryConditions(self,S:float) -> None:
		# Compute DoFs
		sub_space_th=self.Space.sub(0).sub(2)
		sub_space_th_collapsed=sub_space_th.collapse()

		# Grabovski-Berger vortex with final slope
		def grabovski_berger(r) -> np.ndarray:
			psi=(self.r_max-r)/(self.r_max-self.r_phy)/self.r_phy
			mr=r<1
			psi[mr]=r[mr]*(2-r[mr]**2)
			ir=np.logical_and(r>=1,r<self.r_phy)
			psi[ir]=1/r[ir]
			return psi

		class InletAzimuthalVelocity():
			def __init__(self, S): self.S = S
			def __call__(self, x): return self.S*grabovski_berger(x[1]).astype(pet.ScalarType)

		# Modified vortex that goes to zero at top boundary
		self.u_inlet_th=dfx.Function(sub_space_th_collapsed)
		self.inlet_azimuthal_velocity=InletAzimuthalVelocity(S) # Required to smoothly increase S
		self.u_inlet_th.interpolate(self.inlet_azimuthal_velocity)
		self.u_inlet_th.x.scatter_forward()
		
		# Degrees of freedom
		dofs_inlet_th = dfx.fem.locate_dofs_geometrical((sub_space_th, sub_space_th_collapsed), self.inlet)
		bcs_inlet_th = dfx.DirichletBC(self.u_inlet_th, dofs_inlet_th, sub_space_th) # u_th=S*psi(r) at x=0

		# Actual BCs
		_, bcs_inlet_x = self.ConstantBC('x',self.inlet,1) # u_x =1
		self.bcs = [bcs_inlet_x, bcs_inlet_th]			   # x=X entirely handled by implicit Neumann
		
		# Handle homogeneous boundary conditions
		homogeneous_boundaries=[(self.inlet,['r']),(self.top,['r','th']),(self.symmetry,['r','th'])]
		for tup in homogeneous_boundaries:
			marker,directions=tup
			for direction in directions:
				_, bcs=self.ConstantBC(direction,marker)
				self.bcs.append(bcs)
	
	def Baseflow(self,hot_start:bool,save:bool,S:float):
		# Apply new BC
		self.inlet_azimuthal_velocity.S=S
		self.u_inlet_th.interpolate(self.inlet_azimuthal_velocity)
		self.u_inlet_th.x.scatter_forward()
		
		# Memoisation
		if hot_start: self.HotStart(S)
		# Compute form
		base_form  = self.NavierStokes() #no azimuthal decomposition for base flow
		dbase_form = self.LinearisedNavierStokes(0) # m=0

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
	def BaseflowRange(self,Ss) -> None:
		shutil.rmtree(self.dat_real_path)
		shutil.rmtree(self.print_path)
		for S in Ss: 	   # Then on swirl intensity
			if COMM_WORLD.rank==0:
				print("##########################")
				print("Swirl intensity: ",	    S)
			self.Baseflow(True,True,S)
				
		#write result of current mu
		u,p=self.q.split()
		with XDMFFile(COMM_WORLD, self.case_path+"last_u.xdmf", "w") as xdmf:
			xdmf.write_mesh(self.mesh)
			xdmf.write_function(u)
		viewer = pet.Viewer().createMPIIO(self.case_path+"last_baseflow_real.dat", 'w', COMM_WORLD)
		self.q.vector.view(viewer)
		if COMM_WORLD.rank==0: print("Last checkpoint written!")

	def MinimumAxial(self) -> float:
		u,p=self.q.split()
		u=u.compute_point_values()[:,self.direction_map['x']]
		mu=np.min(u)
		mu=COMM_WORLD.reduce(mu,op=MIN) # minimum across processors
		return mu