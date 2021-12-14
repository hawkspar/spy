# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import os, ufl
import numpy as np
import dolfinx as dfx
from spyt import spyt
from dolfinx.io import XDMFFile
from petsc4py import PETSc as pet
from mpi4py.MPI import COMM_WORLD

# Swirling Parallel Yaj Baseflow
class spyb(spyt):
	def __init__(self, meshpath: str, datapath: str, Re: float) -> None:
		super().__init__(meshpath, datapath, Re)
		self.BoundaryConditions(0)
		
	# Memoisation routine - find closest in S
	def HotStart(self,S) -> None:
		closest_file_name=self.datapath+"last_baseflow_real.dat"
		file_names = [f for f in os.listdir(self.datapath+self.baseflow_path+'dat_real/') if f[-3:]=="dat"]
		d=np.infty
		for file_name in file_names:
			Sd = float(file_name[11:16]) # Take advantage of file format 
			fd = abs(S-Sd)#+abs(Re-Red)
			if fd<d: d,closest_file_name=fd,self.datapath+self.baseflow_path+'dat_real/'+file_name
		viewer = pet.Viewer().createMPIIO(closest_file_name, 'r', COMM_WORLD)
		self.q.vector.load(viewer)
		self.q.vector.ghostUpdate(addv=pet.InsertMode.INSERT, mode=pet.ScatterMode.FORWARD)
		print("Loaded "+closest_file_name+" as part of memoisation scheme")

	# Baseflow (really only need DirichletBC objects)
	def BoundaryConditions(self,S:float) -> None:
		# Compute DoFs
		sub_space_th=self.Space.sub(0).sub(2)
		sub_space_th_collapsed=sub_space_th.collapse()

		# Grabovski-Berger vortex with final slope
		def grabovski_berger(r) -> np.ndarray:
			psi=(self.r_max+self.l-r)/self.l/self.r_max
			mr=r<1
			psi[mr]=r[mr]*(2-r[mr]**2)
			ir=np.logical_and(r>=1,r<self.r_max)
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

	def NavierStokes(self) -> ufl.Form:
		r,mu=self.r,self.mu
		rdiv,divr,rgrad,gradr=self.rdiv,self.divr,self.rgrad,self.gradr
		u,p=ufl.split(self.q)
		v,w=ufl.split(self.Test)
		
		# Mass (variational formulation)
		F  = ufl.inner( rdiv(u,0), 	   	 w)
		# Momentum (different test functions and IBP)
		F += ufl.inner(rgrad(u,0)*u,   r*v)      		   # Convection
		F += ufl.inner(rgrad(u,0), gradr(v,0))*mu # Diffusion
		F -= ufl.inner(r*p,		 	divr(v,0)) 		   # Pressure
		return F*ufl.dx
	
	def Baseflow(self,hot_start:bool,save:bool,S:float,nu=1):
		self.mu=nu/self.Re #recalculate viscosity with prefactor
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
		solver.relaxation_parameter=self.rp # Absolutely crucial for convergence
		solver.max_iter=self.max_iter
		solver.rtol=self.rtol
		solver.atol=self.atol
		ksp = solver.krylov_solver
		opts = pet.Options()
		option_prefix = ksp.getOptionsPrefix()
		opts[f"{option_prefix}pc_type"] = "lu"
		opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
		ksp.setFromOptions()
		# Actual heavyweight
		solver.solve(self.q)
		self.q.x.scatter_forward()

		if save:  # Memoisation
			u,p = self.q.split()
			with XDMFFile(COMM_WORLD, self.datapath+self.baseflow_path+"print/u_S="+f"{S:00.3f}"+".xdmf", "w") as xdmf:
				xdmf.write_mesh(self.mesh)
				xdmf.write_function(u)
			viewer = pet.Viewer().createMPIIO(self.datapath+self.baseflow_path+"dat_real/baseflow_S="+f"{S:00.3f}"+".dat", 'w', COMM_WORLD)
			self.q.vector.view(viewer)
			print(".pvd, .dat written!")

	# To be run in real mode
	def BaseflowRange(self,Ss,nus=[1]) -> None:
		for S in Ss: 	# Increase swirl
			for nu in nus: # Decrease viscosity (non physical but helps CV)
				print("viscosity prefactor: ", nu)
				print("swirl intensity: ",	    S)
				self.Baseflow(S>Ss[0],nu==nus[-1],S,nu)
				
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
		u=u.compute_point_values()
		return np.min(u[:,self.direction_map['x']])