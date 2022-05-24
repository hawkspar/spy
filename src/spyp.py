# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
from multiprocessing.sharedctypes import Value
import os, ufl #source /usr/local/bin/dolfinx-complex-mode
import numpy as np
from spy import SPY
import dolfinx as dfx
from dolfinx.io import XDMFFile
from petsc4py import PETSc as pet
from slepc4py import SLEPc as slp
from mpi4py.MPI import COMM_WORLD

p0=COMM_WORLD.rank==0

# Swirling Parallel Yaj Perturbations
class SPYP(SPY):
	def __init__(self, params:dict, datapath:str, Ref, nutf, direction_map:dict, S:float, m:int) -> None:
		super().__init__(params, datapath, Ref, nutf, direction_map)
		self.S=S; self.m=m
		self.save_string='_S='+f"{S:00.3f}"+'_m='+str(m)

	# To be run in complex mode
	def assembleMatrices(self) -> None:
		# Load baseflow
		self.loadStuff(self.S,"last_baseflow.dat",self.dat_complex_path,11,self.q.vector)
		self.nutf(self.S)

		# Complex Jacobian of NS operator
		dform=self.linearisedNavierStokes(self.m)
		self.J = dfx.fem.assemble_matrix(dform)
		self.J.assemble()
		self.J.zeroRowsLocal(self.dofs) # Impose homogeneous BCs

		# Forcing Norm (m*m): here we choose ux^2+ur^2+uth^2 as forcing norm
		u,_ = ufl.split(self.trial)
		v,_ = ufl.split(self.test)
		norm_form=ufl.inner(u,v)*self.r**2*ufl.dx # Same multiplication process as base equations
		self.N = dfx.fem.assemble_matrix(norm_form)
		self.N.assemble()
		#self.N.zeroRowsLocal(self.dofs,0)
		if p0: print("Jacobian & Norm matrices computed !")

		v=dfx.Function(self.TH)
		v.vector.zeroEntries()
		v.vector[self.dofs]=np.ones(self.dofs.size)
		u,_=v.split()
		with XDMFFile(COMM_WORLD, "sanity_check_bcs.xdmf","w") as xdmf:
			xdmf.write_mesh(self.mesh)
			xdmf.write_function(u)

	def resolvent(self,k:int,St_list):
		# Get u subspace
		u_space, _ = self.TH.sub(0).collapse(collapsed_dofs=True)
		u,_ = ufl.split(self.trial)
		v,_ = ufl.split(self.test)
		w = ufl.TrialFunction(u_space)
		z = ufl.TestFunction(u_space)
		# Quadrature-extensor QE (m*n) reshapes forcing vector (n*1) to (m*1) and compensates the quadrature in R.
		QE_form=ufl.inner(w,v)*self.r**2*ufl.dx
		# Mass Mq (m*m): required to have a proper maximisation problem in a cylindrical geometry 
		Mq_form=ufl.inner(u,v)*self.r*ufl.dx#+ufl.inner(p,s)*self.r*ufl.dx # Quadrature corresponds to L2 integration
		# Mass Mf (n*n): same
		Mf_form=ufl.inner(w,z)*self.r*ufl.dx
		# Assembling matrices
		QE = dfx.fem.assemble_matrix(QE_form); QE.assemble()
		Mq = dfx.fem.assemble_matrix(Mq_form); Mq.assemble()
		Mf = dfx.fem.assemble_matrix(Mf_form); Mf.assemble()

		# Sizes
		n=u_space.dofmap.index_map.size_global * u_space.dofmap.index_map_bs
		n_local = Mf.local_size[0]
		if p0: print("Quadrature, Extractor & Mass matrices computed !")

		# Necessary for matrix-free routine
		class LHS_class:
			def mult(self,A,x,y):
				w, z = Mq.createVecs()
				QE.mult(x,w)
				KSPs[0].solve(w,z)
				Mq.mult(z,w)
				KSPs[1].solve(w,z)
				QE.multTranspose(z,y)

		# Solver
		EPS = slp.EPS(); EPS.create()
		for St in St_list:
			L=self.J-1j*np.pi*St*self.N # Equations (Fourier transform is -2j pi f but Strouhal is St=fD/U=2fR/U)

			KSPs = []
			# Useful solvers (here to put options for computing a smart R)
			for Mat in [L,L.hermitianTranspose()]:
				KSP = pet.KSP().create()
				KSP.setOperators(Mat)
				KSP.setTolerances(rtol=self.params['rtol'], atol=self.params['atol'], max_it=self.params['max_iter'])
				# Krylov subspace
				KSP.setType('preonly')
				# Preconditioner
				PC = KSP.getPC(); PC.setType('lu')
				PC.setFactorSolverType('mumps')
				KSP.setFromOptions()
				KSPs.append(KSP)
			
			# Tests
			x=self.mesh.geometry.x
			FE_vector_1=ufl.VectorElement("Lagrange",self.mesh.ufl_cell(),1,3)
			FE_vector_2=ufl.VectorElement("Lagrange",self.mesh.ufl_cell(),2,3)
			V1 = dfx.FunctionSpace(self.mesh,FE_vector_1)
			v1 = dfx.Function(V1)
			v1.vector.zeroEntries()
			v1.vector[::3]=np.sin(np.pi*(x[:,1]-1))*np.exp(-.5*((x[:,0]-2)**2/.2**2+(x[:,1]-1)**2/.2**2))
			V2 = dfx.FunctionSpace(self.mesh,FE_vector_2)
			v2 = dfx.Function(V2)
			v2.interpolate(v1)
			ft = dfx.Function(self.TH)
			ft.vector.zeroEntries()
			_, map_U = self.TH.sub(0).collapse(collapsed_dofs=True)
			ft.vector[map_U]=v2.vector
			ft.vector.assemble()
			ftu,_=ft.split()
			with XDMFFile(COMM_WORLD, "sanity_check_gaussian_input.xdmf","w") as xdmf:
				xdmf.write_mesh(self.mesh)
				xdmf.write_function(ftu)
			qt = dfx.Function(self.TH)
			x, _ = self.N.createVecs()
			self.N.mult(ft.vector,x)
			fx = dfx.Function(self.TH)
			fx.vector[:]=x
			fxu,_=fx.split()
			with XDMFFile(COMM_WORLD, "sanity_check_gaussian_x.xdmf","w") as xdmf:
				xdmf.write_mesh(self.mesh)
				xdmf.write_function(fxu)
			KSPs[0].solve(x,qt.vector)
			ut,_=qt.split()
			with XDMFFile(COMM_WORLD, "sanity_check_gaussian_test.xdmf","w") as xdmf:
				xdmf.write_mesh(self.mesh)
				xdmf.write_function(ut)
			
			# Matrix free operator
			LHS=pet.Mat()
			LHS.create(comm=COMM_WORLD)
			LHS.setSizes([[n_local,n],[n_local,n]])
			LHS.setType(pet.Mat.Type.PYTHON)
			LHS.setPythonContext(LHS_class())
			LHS.setUp()

			# Eigensolver
			EPS.setOperators(LHS,Mf) # Solve E^T*Q^T*L^-1H*Mq*L^-1*Q*E*f=sigma^2*Mf*f (cheaper than a proper SVD)
			EPS.setProblemType(slp.EPS.ProblemType.GHEP) # Specify that A is hermitian (by construction), but M is semi-positive
			EPS.setWhichEigenpairs(EPS.Which.LARGEST_MAGNITUDE) # Find largest eigenvalues
			EPS.setDimensions(k,max(10,2*k)) # Find k eigenvalues only with max number of Lanczos vectors
			EPS.setTolerances(self.params['atol'],self.params['max_iter']) # Set absolute tolerance and number of iterations
			EPS.setFromOptions()
			if p0: print("Solver launch...")
			EPS.solve()
			n=EPS.getConverged()
			if n==0: continue

			# Conversion back into numpy 
			gains=np.sqrt(np.array([EPS.getEigenvalue(i) for i in range(n)],dtype=np.complex))
			#write gains
			if not os.path.isdir(self.resolvent_path): os.mkdir(self.resolvent_path)
			np.savetxt(self.resolvent_path+"gains"+self.save_string+"_St="+f"{St:00.3f}"+".dat",np.abs(gains))

			# Write eigenvectors
			for i in range(min(n,k)):
				# Obtain forcings as eigenvectors
				fu=dfx.Function(self.TH.sub(0).collapse(collapsed_dofs=True)[0])
				EPS.getEigenvector(i,fu.vector)
				fu.x.scatter_forward()
				with XDMFFile(COMM_WORLD, self.resolvent_path+"forcing_u" +self.save_string+"_St="+f"{St:00.3f}"+"_n="+f"{i+1:1d}"+".xdmf","w") as xdmf:
					xdmf.write_mesh(self.mesh)
					xdmf.write_function(fu)

				# Obtain response from forcing
				q=dfx.Function(self.TH)
				_, tmp = QE.createVecs()
				QE.mult(fu.vector,tmp)
				KSP.solve(tmp,q.vector)
				u,p=q.split()
				with XDMFFile(COMM_WORLD, self.resolvent_path+"response_u"+self.save_string+"_St="+f"{St:00.3f}"+"_n="+f"{i+1:1d}"+".xdmf","w") as xdmf:
					xdmf.write_mesh(self.mesh)
					xdmf.write_function(u)
			if p0: print("Strouhal",St,"handled !")

	# Modal analysis
	def eigenvalues(self,sigma:complex,k:int) -> None:
		# Solver
		EPS = slp.EPS(COMM_WORLD); EPS.create()
		EPS.setOperators(-self.J,self.N) # Solve Ax=sigma*Mx
		EPS.setProblemType(slp.EPS.ProblemType.PGNHEP) # Specify that A is no hermitian, but M is semi-positive
		EPS.setWhichEigenpairs(EPS.Which.TARGET_MAGNITUDE) # Find eigenvalues close to sigma
		EPS.setTarget(sigma)
		EPS.setDimensions(k,max(10,2*k)) # Find k eigenvalues only with max number of Lanczos vectors
		EPS.setTolerances(self.params['atol'],self.params['max_iter']) # Set absolute tolerance and number of iterations
		# Spectral transform
		ST  = EPS.getST();  ST.setType('sinvert')
		# Krylov subspace
		KSP = ST.getKSP(); KSP.setType('preonly')
		# Preconditioner
		PC  = KSP.getPC();  PC.setType('lu')
		PC.setFactorSolverType('mumps')
		EPS.setFromOptions()
		EPS.solve()
		n=EPS.getConverged()
		if n==0: return
		# Conversion back into numpy 
		vals=np.array([EPS.getEigenvalue(i) for i in range(n)],dtype=np.complex)
		if not os.path.isdir(self.eig_path): os.mkdir(self.eig_path)
		# write eigenvalues
		np.savetxt(self.eig_path+"evals"+self.save_string+"_sigma="+f"{np.real(sigma):00.3f}"+f"{np.imag(sigma):+00.3f}"+"j.dat",np.column_stack([vals.real, vals.imag]))
		# Write eigenvectors back in xdmf (but not too many of them)
		for i in range(min(n,3)):
			q=dfx.Function(self.TH)
			EPS.getEigenvector(i,q.vector)
			q.x.scatter_forward()
			u,p = q.split()
			with XDMFFile(COMM_WORLD, self.eig_path+"u/evec_u_S="+f"{self.S:00.3f}"+"_m="+str(self.m)+"_lam="+f"{vals[i].real:00.3f}"+f"{vals[i].imag:+00.3f}"+"j.xdmf", "w") as xdmf:
				xdmf.write_mesh(self.mesh)
				xdmf.write_function(u)
		if p0: print("Eigenpairs written !")