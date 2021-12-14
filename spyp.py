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
from slepc4py import SLEPc as slp
from mpi4py.MPI import COMM_WORLD # Here parametric study so mre interesting to have processors doing independant stuff

# Swirling Parallel Yaj Perturbations
class spyp(spyt):
	def __init__(self, meshpath: str, datapath: str, Re: float, S:float, m:int) -> None:
		super().__init__(meshpath, datapath, Re)
		self.resolvent_path	='resolvent/'
		self.eig_path		='eigenvalues/'
		self.S=S; self.m=m
		self.save_string='_S='+f"{S:00.3f}"+'_m='+str(m)
		
	# Memoisation routine - find closest in S
	def LoadBaseflow(self,S) -> None:
		closest_file_name=self.datapath+"last_baseflow_complex.dat"
		file_names = [f for f in os.listdir(self.datapath+self.baseflow_path+'dat_complex/') if f[-3:]=="dat"]
		d=np.infty
		for file_name in file_names:
			Sd = float(file_name[11:16]) # Take advantage of file format 
			fd = abs(S-Sd)#+abs(Re-Red)
			if fd<d: d,closest_file_name=fd,self.datapath+self.baseflow_path+'dat_complex/'+file_name
		viewer = pet.Viewer().createMPIIO(closest_file_name, 'r', COMM_WORLD)
		self.q.vector.load(viewer)
		self.q.vector.ghostUpdate(addv=pet.InsertMode.INSERT, mode=pet.ScatterMode.FORWARD)
		print("Loaded "+closest_file_name+" as part of memoisation scheme")

	# Perturbations (really only need dofs)
	def BoundaryConditionsPerturbations(self,m) -> None:
		# Handle homogeneous boundary conditions
		homogeneous_boundaries=[(self.inlet,['x','r','th']),(self.top,['r','th'])]
		if 	     m ==0: homogeneous_boundaries.append((self.symmetry,['r','th']))
		elif abs(m)==1: homogeneous_boundaries.append((self.symmetry,['x']))
		else:		    homogeneous_boundaries.append((self.symmetry,['x','r','th']))
		self.dofps = np.empty(0,dtype=np.int32)
		for tup in homogeneous_boundaries:
			marker,directions=tup
			for direction in directions:
				dofs, _=self.ConstantBC(direction,marker)
				self.dofps=np.union1d(self.dofps,dofs)

	# To be run in complex mode
	def AssembleMatrices(self) -> None:
		# Load baseflow
		self.LoadBaseflow(self.S)
		
		# Computation of boundary condition dofs (only homogenous enforced, great for perturbations)
		self.BoundaryConditionsPerturbations(self.m)

		# Complex Jacobian of NS operator
		dform=self.LinearisedNavierStokes(self.m)
		self.J = dfx.fem.assemble_matrix(dform)
		self.J.assemble()
		self.J.zeroRowsColumnsLocal(self.dofps) # Impose homogeneous BCs

		# Forcing norm M (m*m): here we choose ux^2+ur^2+uth^2 as forcing norm
		u,p = ufl.split(self.Trial)
		v,w = ufl.split(self.Test)
		form=ufl.inner(u,v)*self.r**2*ufl.dx # Same multiplication process as momentum equations
		self.N = dfx.fem.assemble_matrix(form)
		self.N.assemble()
		self.N.zeroRowsColumnsLocal(self.dofps,0)

		print("Matrices computed !")
	
	def Resolvent(self,k:int,freq_list):
		U,p=self.q.split()
		u,v,w=U.split()
		u=u.compute_point_values()
		print("check base flow max and min in u:",np.max(u),",",np.min(u))

		#quadrature Q (m*m): it is required to compensate the quadrature in resolvent operator R, because R=(A-i*omegaB)^(-1)
		u,p = ufl.split(self.Trial)
		v,w = ufl.split(self.Test)
		Q_form=ufl.dot(u,v)*self.r**2*ufl.dx+ufl.inner(p,w)*self.r*ufl.dx
		Q = dfx.fem.assemble_matrix(Q_form)
		Q.assemble()

		#matrix B (m*n) reshapes forcing vector (n*1) to (m*1). In principal, it contains only 0 and 1 elements. It's essentially an extensor.
		#It can also restrict the flow regions of forcing, to be implemented later. 
		#A note for very rare case: if one wants to damp but not entirely eliminate the forcing in some regions (not the case here), one can put values between 0 and 1. In that case, the matrix I in the following is not the same as P. 
		
		row_indices=self.Space.sub(0).dofmap().dofs() #get all index related to u
		row_indices.sort()
		m=len(self.Space.dofmap().dofs())
		n=len(row_indices)
		B=pet.Mat().createAIJ([m, n]) # AIJ represents sparse matrix
		B.setUp()
		B.assemble()
		for i in range(n): B.setValue(row_indices[i],i,1)

		# Necessary for matrix-free routine
		class LHS_class:
			def __init__(self,N,L,Q,B):
				self.N=N
				self.L=L
				self.Q=Q
				self.B=B
			
			def mult(self,A,x,y):
				tmp=pet.Vec().createSeq(m)
				self.B.mult(x,y)
				self.Q.mult(y,tmp)
				self.L.solve(tmp,y)
				self.N.mult(y,tmp)
				self.L.solveTranspose(tmp,y)
				self.Q.multTranspose(y,tmp)
				self.B.multTranspose(tmp,y)

		# Solver
		E = slp.EPS(); E.create()
		for freq in freq_list:
			L=self.J-2j*np.pi*freq*self.N # Equations

			# Matrix free operator
			LHS=pet.Mat()
			LHS.create(comm=COMM_WORLD)
			LHS.setSizes([n,n])
			LHS.setPythonContext(pet.Mat.Type.PYTHON)
			LHS.setPythonContext(LHS_class(self.N,L,Q,B))
			LHS.setUp()

			# Eigensolver
			E.setOperators(LHS) # Solve Rx=sigma*x (cheaper than a proper SVD)
			E.setWhichEigenpairs(E.Which.LARGEST_MAGNITUDE) # Find eigenvalues close to sigma
			E.setDimensions(k,max(10,2*k)) # Find k eigenvalues only with max number of Lanczos vectors
			E.setTolerances(self.atol,100) # Set absolute tolerance and number of iterations
			E.setProblemType(slp.EPS.ProblemType.NHEP) # Specify that A is no hermitian, but M is semi-positive
			# Spectral transform
			ST  = E.getST()
			# Krylov subspace
			KSP = ST.getKSP(); KSP.setType('preonly')
			# Preconditioner
			PC  = KSP.getPC();  PC.setType('lu')
			PC.setFactorSolverType('mumps')
			E.setFromOptions()
			E.solve()
			n=E.getConverged()
			if n==0: continue
			# Conversion back into numpy 
			gains=np.array([E.getEigenvalue(i) for i in range(n)],dtype=np.complex)
			
			#write gains
			np.savetxt(self.datapath+self.resolvent_path+"gains"+self.save_string+"f="+f"{freq:00.3f}"+".dat",np.abs(gains))
			# Write eigenvectors back in pvd (but not too many of them)
			for i in range(min(n,3)):
				# Obtain forcings as eigenvectors
				f=dfx.Function(self.Space)
				fu,fp = f.split()
				E.getEigenvector(i,fu.vector)
				fu.x.scatter_forward()
				with XDMFFile(COMM_WORLD, self.datapath+self.resolvent_path+"forcing_u" +self.save_string+"f="+f"{freq:00.3f}"+"_n="+f"{i+1:1d}"+".xdmf","w") as xdmf:
					xdmf.parameters["flush_output"] = True
					xdmf.write_mesh(self.mesh)
					xdmf.write_function(fu)

				# Obtain response from forcing
				q=dfx.Function(self.Space)
				u,p=q.split()
				tmp=pet.Vec().createSeq(m)
				B.mult(fu.vector,u.vector)
				Q.mult(u.vector,tmp)
				L.solve(tmp,u.vector)
				with XDMFFile(COMM_WORLD, self.datapath+self.resolvent_path+"response_u" +self.save_string+"f="+f"{freq:00.3f}"+"_n="+f"{i+1:1d}"+".xdmf","w") as xdmf:
					xdmf.write_mesh(self.mesh)
					xdmf.write_function(u)

	def Eigenvalues(self,sigma:complex,k:int) -> None:
		# Solver
		E = slp.EPS(COMM_WORLD); E.create()
		E.setOperators(-self.J,self.N) # Solve Ax=sigma*Mx
		E.setWhichEigenpairs(E.Which.TARGET_MAGNITUDE) # Find eigenvalues close to sigma
		E.setTarget(sigma)
		E.setDimensions(k,max(10,2*k)) # Find k eigenvalues only with max number of Lanczos vectors
		#E.setTolerances(eps,60) # Set absolute tolerance and number of iterations
		E.setTolerances(1e-2,10)
		E.setProblemType(slp.EPS.ProblemType.PGNHEP) # Specify that A is no hermitian, but M is semi-positive
		# Spectral transform
		ST  = E.getST();    ST.setType('sinvert')
		# Krylov subspace
		KSP = ST.getKSP(); KSP.setType('preonly')
		# Preconditioner
		PC  = KSP.getPC();  PC.setType('lu')
		PC.setFactorSolverType('mumps')
		E.setFromOptions()
		E.solve()
		n=E.getConverged()
		if n==0: return
		# Conversion back into numpy 
		vals=np.array([E.getEigenvalue(i) for i in range(n)],dtype=np.complex)
		# write eigenvalues
		np.savetxt(self.datapath+self.eig_path+"evals"+self.save_string+"_sigma="+f"{np.real(sigma):00.3f}"+f"{np.imag(sigma):+00.3f}"+"j.dat",np.column_stack([vals.real, vals.imag]))
		# Write eigenvectors back in xdmf (but not too many of them)
		for i in range(min(n,3)):
			q=dfx.Function(self.Space)
			E.getEigenvector(i,q.vector)
			q.x.scatter_forward()
			u,p = q.split()
			with XDMFFile(COMM_WORLD, self.datapath+self.eig_path+"u/evec_u_S="+f"{self.S:00.3f}"+"_m="+str(self.m)+"_lam="+f"{vals[i].real:00.3f}"+f"{vals[i].imag:+00.3f}"+"j.xdmf", "w") as xdmf:
				xdmf.write_mesh(self.mesh)
				xdmf.write_function(u)
		print("Eigenpairs written !")