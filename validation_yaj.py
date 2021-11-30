# coding: utf-8
"""
Created on Wed Oct 13 13:50:00 2021

@author: hawkspar
"""
import os, ufl
import numpy as np
import dolfinx as dfx
from pdb import set_trace
import scipy.sparse as sps
from scipy.io import savemat
import scipy.sparse.linalg as la
#from petsc4py import PETSc as pet
from mpi4py.MPI import COMM_WORLD
from dolfinx.io import XDMFFile, VTKFile
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence

rp =.99 #relaxation_parameter
ae =1e-9 #absolute_tolerance
eps=1e-14 #DOLFIN_EPS does not work well
direction_map={'x':0,'r':1,'th':2}

# Geometry parameters
x_max=70; r_max=10; l=50

# Jet geometry
def inlet(	 x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[0],0,	  	eps) # Left border
def symmetry(x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[1],0,	  	eps) # Axis of symmetry at r=0
def outlet(	 x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[0],x_max+l,eps) # Right border
def top(	 x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[1],r_max+l,eps) # Top boundary at r=R

# Gradient with x[0] is x, x[1] is r, x[2] is theta
def grad(v,r,m):
	return ufl.as_tensor([[v[0].dx(0), v[0].dx(1), m*1j*v[0]/r],
				 	  	  [v[1].dx(0), v[1].dx(1), m*1j*v[1]/r-v[2]/r],
					  	  [v[2].dx(0), v[2].dx(1), m*1j*v[2]/r+v[1]/r]])

def rgrad(v,r,m):
	return ufl.as_tensor([[r*v[0].dx(0), r*v[0].dx(1), m*1j*v[0]],
				 	  	  [r*v[1].dx(0), r*v[1].dx(1), m*1j*v[1]-v[2]],
					  	  [r*v[2].dx(0), r*v[2].dx(1), m*1j*v[2]+v[1]]])

def gradr(v,r,m):
	return ufl.as_tensor([[r*v[0].dx(0), v[0]+r*v[0].dx(1), m*1j*v[0]],
				 	  	  [r*v[1].dx(0), v[1]+r*v[1].dx(1), m*1j*v[1]-v[2]],
					  	  [r*v[2].dx(0), v[2]+r*v[2].dx(1), m*1j*v[2]+v[1]]])

# Same for divergent
def div(v,r,m):  return   v[0].dx(0) + (r*v[1]).dx(1)/r 	  + m*1j*v[2]/r

def rdiv(v,r,m): return r*v[0].dx(0) +    v[1] + r*v[1].dx(1) + m*1j*v[2]

def divr(v,r,m): return r*v[0].dx(0) +  2*v[1] + r*v[1].dx(1) + m*1j*v[2]

# Grabovski-Berger vortex with final slope
def grabovski_berger(r) -> np.ndarray:
	psi=(r_max+l-r)/l/r_max
	mr=r<1
	psi[mr]=r[mr]*(2-r[mr]**2)
	ir=np.logical_and(r>=1,r<r_max)
	psi[ir]=1/r[ir]
	return psi

class InletAzimuthalVelocity():
	def __init__(self, S): self.S = S
	def __call__(self, x): return self.S*grabovski_berger(x[1])

def csi(a,b): return .5*(1+np.tanh(4*np.tan(-np.pi/2+np.pi*np.abs(a-b)/l)))

def Ref(x,Re_s,Re):
	Rem=np.ones(x[0].size)*Re
	x_ext=x[0]>x_max
	Rem[x_ext]=Re		 +(Re_s-Re) 	   *csi(np.minimum(x[0][x_ext],x_max+l),x_max) # min necessary to prevent spurious jumps because of mesh conversion
	r_ext=x[1]>r_max
	Rem[r_ext]=Rem[r_ext]+(Re_s-Rem[r_ext])*csi(np.minimum(x[1][r_ext],r_max+l),r_max)
	return Rem

# Sponged Reynolds number
def sponged_Reynolds(Re,mesh) -> dfx.Function:
	Re_s=.1
	Red=dfx.Function(dfx.FunctionSpace(mesh,ufl.FiniteElement("Lagrange",mesh.ufl_cell(),2)))
	Red.interpolate(lambda x: Ref(x,Re_s,Re))
	return Red

# Helper functions for sparse matrices
def csr_zero_rows(csr : sps.csr_matrix, rows : np.ndarray):
	for row in rows: csr.data[csr.indptr[int(row)]:csr.indptr[int(row)+1]] = 0

def csc_zero_cols(csc : sps.csc_matrix, cols : np.ndarray):
	for col in cols: csc.data[csc.indptr[int(col)]:csc.indptr[int(col)+1]] = 0

class yaj():
	def __init__(self,meshpath:str,datapath:str,m:int,Re:float,S:float,n_S:int):
		# Newton solver
		self.mu	 =1   # viscosity prefator
		self.n_nu=1   # number of viscosity iterations
		self.S   =S   # swirl amplitude relative to main flow
		self.n_S =n_S # number of swirl iterations
		self.m   =m   # azimuthal decomposition

		# Paths
		self.datapath = datapath
		self.private_path  	='doing/'
		self.resolvent_path	='resolvent/'
		self.eig_path		='eigenvalues/'
		#Re_string		='_Re='+str(Re)+'_'
		baseflow_string ='_S='+f"{S:00.3f}"
		self.save_string=baseflow_string+'_m='+str(m)
		
		# Mesh from file
		with XDMFFile(COMM_WORLD, meshpath, "r") as file:
			self.mesh = file.read_mesh(name="Grid")
		
		# Taylor Hodd elements ; stable element pair
		FE_vector=ufl.VectorElement("Lagrange",self.mesh.ufl_cell(),2,3)
		FE_scalar=ufl.FiniteElement("Lagrange",self.mesh.ufl_cell(),1)
		self.Space=dfx.FunctionSpace(self.mesh,FE_vector*FE_scalar)	# full vector function space
		
		# Test & trial functions
		self.Test  = ufl.TestFunction(self.Space)
		self.Trial = ufl.TrialFunction(self.Space)
		
		# Extraction of r and Re computation
		self.r = ufl.SpatialCoordinate(self.mesh)[1]
		self.Re = sponged_Reynolds(Re,self.mesh)
		self.mu = 1/self.Re
		self.q = dfx.Function(self.Space) # Initialisation of q
		
	# Memoisation routine - find closest in Re and S
	def HotStart(self,S) -> None:
		closest_file_name=self.datapath+"last_baseflow.npy"
		file_names = [f for f in os.listdir(self.datapath+self.private_path+'dat/') if f[-3:]=="npy"]
		d=np.infty
		for file_name in file_names:
			Sd = float(file_name[11:16]) # Take advantage of file format 
			fd = abs(S-Sd)#+abs(Re-Red)
			if fd<d: d,closest_file_name=fd,self.datapath+self.private_path+'dat/'+file_name
		#self.q.vector.ghostUpdate(addv=pet.InsertMode.INSERT, mode=pet.ScatterMode.FORWARD)
		self.q.x.array.real=np.load(closest_file_name,allow_pickle=True)
		
	# Code factorisation
	def ConstantBC(self, direction:chr, boundary, value=0) -> tuple:
		sub_space=self.Space.sub(0).sub(direction_map[direction])
		sub_space_collapsed=sub_space.collapse()
		# Compute proper zeros
		constant=dfx.Function(sub_space_collapsed)
		with constant.vector.localForm() as zero_loc: zero_loc.set(value)
		# Compute DoFs
		dofs = dfx.fem.locate_dofs_geometrical((sub_space, sub_space_collapsed), boundary)
		# Actual BCs
		bcs = dfx.DirichletBC(constant, dofs, sub_space) # u_i=value at boundary
		return dofs[0], bcs # Only return unflattened dofs

	# Baseflow (really only need DirichletBC objects)
	def BoundaryConditions(self,S:float) -> None:
		# Compute DoFs
		sub_space_th=self.Space.sub(0).sub(2)
		sub_space_th_collapsed=sub_space_th.collapse()

		# Modified vortex that goes to zero at top boundary
		u_inlet_th=dfx.Function(sub_space_th_collapsed)
		self.inlet_azimuthal_velocity=InletAzimuthalVelocity(S) # Required to smoothly increase S
		u_inlet_th.interpolate(self.inlet_azimuthal_velocity)
		u_inlet_th.x.scatter_forward()
		
		# Degrees of freedom
		dofs_inlet_th = dfx.fem.locate_dofs_geometrical((sub_space_th, sub_space_th_collapsed), inlet)
		bcs_inlet_th = dfx.DirichletBC(u_inlet_th, dofs_inlet_th, sub_space_th) # u_th=S*psi(r) at x=0

		# Actual BCs
		dofs_inlet_x, bcs_inlet_x = self.ConstantBC('x',inlet,1) # u_x =1
		self.bcs = [bcs_inlet_x, bcs_inlet_th]									# x=X entirely handled by implicit Neumann
		
		# Handle homogeneous boundary conditions
		homogeneous_boundaries_dic={'inlet':['r'],'top':['r','th'],'symmetry':['r','th']}
		for boundary in homogeneous_boundaries_dic:
			for direction in homogeneous_boundaries_dic[boundary]:
				_, bcs=self.ConstantBC(direction,lambda x: eval(boundary+'(x)'))
				self.bcs.append(bcs)

	# Perturbations (really only need dofs)
	def BoundaryConditionsPerturbations(self) -> None:
		# Handle homogeneous boundary conditions
		homogeneous_boundaries_dic={'inlet':['x','r','th'],'top':['r','th']}
		if 	     self.m ==0: homogeneous_boundaries_dic['symmetry']=['r','th']
		elif abs(self.m)==1: homogeneous_boundaries_dic['symmetry']=['x']
		else:				 homogeneous_boundaries_dic['symmetry']=['x','r','th']
		self.bcps = []; self.dofps = np.empty(0)
		for boundary in homogeneous_boundaries_dic:
			for direction in homogeneous_boundaries_dic[boundary]:
				dofs, bcs=self.ConstantBC(direction,lambda x: eval(boundary+'(x)'))
				self.bcps.append(bcs)
				self.dofps=np.union1d(self.dofps,dofs)

	def NonlinearOperator(self) -> ufl.Form:
		r=self.r
		u,p=ufl.split(self.q)
		v,w=ufl.split(self.Test)
		
		# Mass (variational formulation)
		F  = ufl.inner( rdiv(u,r,0), 	   w)
		# Momentum (different test functions and IBP)
		F += ufl.inner(rgrad(u,r,0)*u,   r*v)      		   # Convection
		F += ufl.inner(rgrad(u,r,0), gradr(v,r,0))*self.mu # Diffusion
		F -= ufl.inner(r*p,			  divr(v,r,0)) 		   # Pressure
		return F*ufl.dx

	def MassTerm(self,m:int) -> ufl.Form:
		r=self.r
		u_b,p_b=ufl.split(self.q) # Baseflow
		v,	w  =ufl.split(self.Test)
		
		# Mass (variational formulation)
		return ufl.inner(rdiv(u_b,r,m), w)*ufl.dx

	def JacobianOperator(self,m:int) -> ufl.Form:
		r=self.r
		u,	p  =ufl.split(self.Trial)
		u_b,p_b=ufl.split(self.q) # Baseflow
		v,	w  =ufl.split(self.Test)
		
		# Mass (variational formulation)
		F  = ufl.inner( rdiv(u,  r,m), 		 w)
		# Momentum (different test functions and IBP)
		F += ufl.inner(rgrad(u_b,r,0)*u,   r*v)    		  	 # Convection
		F += ufl.inner(rgrad(u,  r,m)*u_b, r*v)
		F += ufl.inner(rgrad(u,  r,m), gradr(v,r,m))/self.Re # Diffusion
		F -= ufl.inner(r*p,			    divr(v,r,m)) 		 # Pressure
		return F*ufl.dx

	# To be run in real mode
	def Newton(self,hotstart:bool) -> None:
		if self.n_S>1: Ss= np.cos(np.pi*np.linspace(self.n_S,0,self.n_S)/2/self.n_S)*self.S # Chebychev spacing
		else: 		   Ss=[self.S]
		self.BoundaryConditions(Ss[0]) # Initialises boundary condition
		for S_current in Ss: 	# Increase swirl
			for nu_current in np.linspace(self.mu,1,self.n_nu): # Decrease viscosity (non physical but helps CV)
				print("viscosity prefactor: ", nu_current)
				print("swirl intensity: ",	    S_current)
				self.mu=nu_current/self.Re #recalculate viscosity with prefactor
				self.inlet_azimuthal_velocity.S=S_current
				if hotstart: self.HotStart(S_current) # Memoisation
				# Compute form
				base_form  = self.NonlinearOperator() #no azimuthal decomposition for base flow
				dbase_form = self.JacobianOperator(0)
				problem = dfx.fem.NonlinearProblem(base_form,self.q,bcs=self.bcs,J=dbase_form)
				solver  = dfx.NewtonSolver(COMM_WORLD, problem)
				solver.rtol=eps
				solver.relaxation_parameter=rp # Absolutely crucial for convergence
				solver.max_iter=30
				solver.atol=ae
				#self.q.vector.ghostUpdate(addv=pet.InsertMode.INSERT, mode=pet.ScatterMode.FORWARD)
				solver.solve(self.q)
				if np.isclose(nu_current,1,eps):  # Memoisation
					u,p = self.q.split()
					with VTKFile(COMM_WORLD, self.datapath+self.private_path+"print/u_S="+f"{S_current:00.3f}"+".pvd","w") as vtk:
						vtk.write([u._cpp_object])
					np.save(self.datapath+self.private_path+"dat/baseflow_S="+f"{S_current:00.3f}"+".npy",self.q.x.array.real)
					print(".pvd, .npy written!")
				
		#write result of current mu
		with VTKFile( COMM_WORLD, self.datapath+"last_u.pvd","w") as vtk:
			vtk.write([u._cpp_object])
		np.save(self.datapath+"last_baseflow.npy",self.q.x.array.real)
		print("Last checkpoint written!")

	def ComputeAM(self) -> None:
		# Load baseflow
		self.HotStart(self.S)
		
		# Computation of boundary condition dofs (only homogenous enforced, great for perturbations)
		self.BoundaryConditionsPerturbations()

		# Complex Jacobian of NS operator
		dform=self.JacobianOperator(self.m)
		self.J = dfx.fem.assemble_matrix(dform)
		self.J.assemble()
		
		# Convert from PeTSc to Scipy for easy slicing
		ai, aj, av = self.J.getValuesCSR()
		self.J = sps.csr_matrix((av, aj, ai),dtype=np.complex64)
		# Efficiently cancel out rows and columns
		csr_zero_rows(self.J,self.dofps)
		self.J=self.J.tocsc()
		csc_zero_cols(self.J,self.dofps)
		# Introduce a -1 to force homogeneous BC
		self.J[self.dofps,self.dofps]=-1
		self.J.eliminate_zeros()

		# Forcing norm M (m*m): here we choose ux^2+ur^2+uth^2 as forcing norm
		u,p   = ufl.split(self.Trial)
		v,phi = ufl.split(self.Test)
		form=ufl.inner(u,v)*self.r**2*ufl.dx # Same multiplication process as base equations
		self.N = dfx.fem.assemble_matrix(form)
		self.N.assemble()

		# Convert too
		mi, mj, mv = self.N.getValuesCSR()
		self.N = sps.csr_matrix((mv, mj, mi),dtype=np.complex64)
		# Efficiently cancel out stuff
		csr_zero_rows(self.N,self.dofps)
		self.N=self.N.tocsc()
		csc_zero_cols(self.N,self.dofps)
		self.N.eliminate_zeros()
		
		# Matlab pipeline
		mdic = {"A": self.J, "M": self.N}
		savemat("Matlab/AM"+self.save_string+".mat", mdic)
		print("Matrices assembled and saved to .mat file")

	def Getw0(self) -> float:
		U,p=self.q.split()
		u,v,w=U.split()
		return np.min(np.abs(u.compute_point_values()))

	def Resolvent(self,k:int,freq_list):
		print("check base flow max and min in u:",np.max(self.q.vector()[:]),",",np.min(self.q.vector()[:]))

		#matrix B (m*m): with matrix A form altogether the resolvent operator
		up=ufl.as_vector((self.Trial[0],self.Trial[1],self.Trial[2]))
		pp=self.Trial[3]
		B_form=ufl.dot(up,self.Test[0])*self.r*ufl.dx
		Ba = dfx.fem.assemble_matrix(B_form)
		Ba.assemble().tocsc()

		#response norm Mr (m*m): here we choose the same as forcing norm
		Mr, Mf = self.N, self.N

		#quadrature Q (m*m): it is required to compensate the quadrature in resolvent operator R, because R=(A-i*omegaB)^(-1)
		Q_form=ufl.dot(up,self.Test[0])*self.r*ufl.dx+pp*self.Test[1]*self.r*ufl.dx
		Qa = dfx.fem.assemble_matrix(Q_form)
		Qa.assemble().tocsc()

		#matrix P (m*n) reshapes forcing vector (n*1) to (m*1). In principal, it contains only 0 and 1 elements.
		#It can also restrict the flow regions of forcing, to be implemented later. 
		#A note for very rare case: if one wants to damp but not entirely eliminate the forcing in some regions (not the case here), one can put values between 0 and 1. In that case, the matrix I in the following is not the same as P. 
		
		index_forcing=self.Space.sub(0).dofmap().dofs() #get all index related to u
		index_forcing.sort()

		row_ind=np.intersect1d(self.self.freeinds,index_forcing) #get free index related to u
		row_ind.sort()
		m=len(self.Space.dofmap().dofs())
		n=len(row_ind)
		col_ind=np.arange(n)
		Pa=sps.csc_matrix((np.ones(n),(row_ind,col_ind)),(m,n))

		#matrix I (m*n) reshapes forcing matrix Mf (m*m) to I^T*Mf*I (n*n). The matrix I can be different from P in that very rare case remarked above.
		Ia=Pa

		B  = Ba[self.self.freeinds,:][:,self.self.freeinds]
		P  = Pa[self.self.freeinds]
		I  = Ia[self.self.freeinds]
		Q  = Qa[self.self.freeinds,:][:,self.self.freeinds]

		Q_shape=np.shape(Q)
		print('matrix Q size: '+str(Q_shape))
		P_shape=np.shape(P)
		print('matrix P size: '+str(P_shape))

		for freq in freq_list:
			R = la.splu(-self.J-2*np.pi*1j*freq*B,permc_spec=3)
			# get response linear operator P^H*Q^H*R^H*Mr*R*Q*P
			def lhs(f): return P.transpose()*Q.transpose()*R.solve(Mr*R.solve(Q*P*f),trans='H')

			LHS = la.LinearOperator((min(P_shape),min(P_shape)),matvec=lhs,dtype='complex')

			# forcing linear operator is on the rhs M=I.transpose()*Mf*I
			gains,eigenvectors = la.eigs(LHS, k=k, M=I.transpose()*Mf*I, sigma=None,  maxiter=100, tol=ae, return_eigenvectors=True)
			
			#write forcing and response
			f=eigenvectors
			r=R.solve(Q*P*f)

			ua = dfx.Function(self.Space) #declaration for efficiency

			for i in range(k):
				ua.vector()[self.self.freeinds] = np.abs(P*f[:,i])
				u,p  = ua.split()
				with VTKFile(COMM_WORLD, self.datapath+self.resolvent_path+"forcing_u" +self.save_string+"f="+f"{freq:00.3f}"+"_n="+f"{i+1:1d}"+".pvd","w") as vtk:
					vtk.write([u._cpp_object])
				ua.vector()[self.self.freeinds] = np.abs(r[:,i])
				u,p  = ua.split()
				with VTKFile(COMM_WORLD, self.datapath+self.resolvent_path+"response_u"+self.save_string+"f="+f"{freq:00.3f}"+"_n="+f"{i+1:1d}"+".pvd","w") as vtk:
					vtk.write([u._cpp_object])
			
			#write gains
			np.savetxt(self.datapath+self.resolvent_path+"gains"+self.save_string+"f="+f"{freq:00.3f}"+".dat",np.real(gains))

	def Eigenvalues(self,sigma:complex,k:int) -> None:
		print("Computing eigenvalues/vectors in Python!")
		ncv = np.max([10,2*k])
		try:
			vals, vecs = la.eigs(-self.J, k=k, M=self.N, sigma=sigma, maxiter=20, tol=1e-3, ncv=ncv)
		except ArpackNoConvergence as err:
			print("Solver not fully converged")
			vals, vecs = err.eigenvalues, err.eigenvectors
			if vals.size==0: return
		# write eigenvalues
		np.savetxt(self.datapath+self.eig_path+"evals"+self.save_string+"_sigma="+f"{np.real(sigma):00.3f}"+f"{np.imag(sigma):+00.3f}"+"j.dat",np.column_stack([vals.real, vals.imag]))
		# Write eigenvectors back in pvd
		for i in range(vals.size):
			q=dfx.Function(self.Space)
			q.vector.array = vecs[:,i]
			u,p = q.split()
			with VTKFile(COMM_WORLD, self.datapath+self.eig_path+"u/evec_u_S="+f"{self.S:00.3f}"+"_m="+str(self.m)+"_lam="+f"{vals[i].real:00.3f}"+f"{vals[i].imag:+00.3f}"+"j.pvd","w") as vtk:
				vtk.write([u._cpp_object])
			with VTKFile(COMM_WORLD, self.datapath+self.eig_path+"p/evec_p_S="+f"{self.S:00.3f}"+"_m="+str(self.m)+"_lam="+f"{vals[i].real:00.3f}"+f"{vals[i].imag:+00.3f}"+"j.pvd","w") as vtk:
				vtk.write([p._cpp_object])