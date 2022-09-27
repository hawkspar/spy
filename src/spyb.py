# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import shutil, ufl
import numpy as np
from spy import SPY, dirCreator
from petsc4py import PETSc as pet
import dolfinx as dfx
from dolfinx.fem import Function, FunctionSpace
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py.MPI import COMM_WORLD as comm, MIN

pet.Options().setValue('-snes_linesearch_type', 'basic') # classical Newton method

p0=comm.rank==0

# Swirling Parallel Yaj Baseflow
class SPYB(SPY):
	def __init__(self, params:dict, datapath:str, Ref, nutf, direction_map:dict, InletAzimuthalVelocity, C:bool=False, full_TH:bool=True) -> None:
		super().__init__(params, datapath, Ref, nutf, direction_map, C)
		if full_TH: th_space,_=self.u_space.sub(direction_map['th']).collapse()
		else:
			FE_vector = ufl.VectorElement("CG",self.mesh.ufl_cell(),2,3)
			V = FunctionSpace(self.mesh,FE_vector)
			th_space,_=V.sub(direction_map['th']).collapse()

		# Modified vortex that goes to zero at top boundary
		self.u_inlet_th=Function(th_space)
		self.inlet_azimuthal_velocity=InletAzimuthalVelocity(0)
		dirCreator(self.baseflow_path)

	def smoothen(self,e):
		FE_scalar = ufl.FiniteElement("CG",self.mesh.ufl_cell(),1)
		V = FunctionSpace(self.mesh,FE_scalar)
		u,v=ufl.TrialFunction(V),ufl.TestFunction(V)
		P,r=self.P,self.r
		grd,div=lambda v: self.grd_nor(v,0),lambda v: self.div_nor(v,0)
		a=ufl.inner(u,v)
		a+=e*ufl.inner(grd(u),grd(v))
		L=ufl.inner(P,v)
		pb = dfx.fem.petsc.LinearProblem(a*r*ufl.dx, L*r*ufl.dx, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
		self.P = pb.solve()
		if p0: print("Smoothing finished !",flush=True)
	
	# Careful here Re is only for printing purposes ; self.Re is a more involved function
	def baseflow(self,Re:int,S:float,hot_start:bool,save:bool=True,baseflowInit=None):
		# Apply new BC
		self.inlet_azimuthal_velocity.S=S
		self.u_inlet_th.interpolate(self.inlet_azimuthal_velocity)
		# Cold initialisation
		if baseflowInit!=None: self.U.interpolate(baseflowInit)
		# Memoisation
		elif hot_start:	self.loadBaseflow(S,Re,True)

		self.Q = Function(self.TH)
		_, V_to_TH = self.TH.sub(0).collapse()
		_, W_to_TH = self.TH.sub(1).collapse()
		self.Q.x.array[V_to_TH] = self.U.x.array
		self.Q.x.array[W_to_TH] = self.P.x.array

		# Compute form
		base_form  = self.navierStokes() #no azimuthal decomposition for base flow
		#dbase_form = self.linearisedNavierStokes(0) # m=0
		
		# Encapsulations
		problem = NonlinearProblem(base_form,self.Q,bcs=self.bcs)#,J=dbase_form)
		solver  = NewtonSolver(comm, problem)

		if p0: print("self.U.size=",self.U.x.array.size,flush=True)
		
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
		if p0: print("Solver launch...",flush=True)
		# Actual heavyweight
		solver.solve(self.Q)

		if save:  # Memoisation
			self.printStuff(self.print_path,f"u_S={S:00.3f}_Re={Re:d}",self.U)
			self.saveBaseflow("_S={S:00.3f}_Re={Re:d}")
			if p0: print(".xmdf, .dat written!",flush=True)

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