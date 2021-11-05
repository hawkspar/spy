# coding: utf-8
"""
Created on Wed Oct 13 13:50:00 2021

@author: hawkspar
"""
import numpy as np
import os, dolfinx, ufl
from petsc4py import PETSc
from pdb import set_trace
import scipy.sparse as sps
import scipy.sparse.linalg as la
from mpi4py.MPI import COMM_WORLD
from dolfinx.io import XDMFFile, VTKFile

rp =.99 #relaxation_parameter
ae =1e-9 #absolute_tolerance
eps=1e-14 #DOLFIN_EPS does not work well

# Geometry parameters
x_max=70
r_max=10
l=50

def extract_up(q,shift=0): return ufl.as_vector((q[shift],q[shift+1],q[shift+2])),q[shift+3]
	
# Jet geometry
def inlet(	 x): return np.isclose(x[0],0,		eps) # Left border
def symmetry(x): return np.isclose(x[1],0,		eps) # Axis of symmetry at r=0
def outlet(	 x): return np.isclose(x[0],x_max+l,eps) # Right border
def misc(	 x): return np.isclose(x[1],r_max+l,eps) # Top boundary

# Gradient with x[0] is x, x[1] is r, x[2] is theta
def grad(v,r,m):
	return ufl.as_tensor([[v[0].dx(0), v[0].dx(1), m*1j*v[0]/r],
				 	  [v[1].dx(0), v[1].dx(1), m*1j*v[1]/r-v[2]/r],
					  [v[2].dx(0), v[2].dx(1), m*1j*v[2]/r+v[1]/r]])

def div(v,r,m): return v[0].dx(0) + (r*v[1]).dx(1)/r + m*1j*v[2]/r

def grabovski_berger(r):
	psi=(r_max+l-r)/l/r_max
	mr=r<1
	psi[mr]=r[mr]*(2-r[mr]**2)
	ir=np.logical_and(r>=1,r<r_max)
	psi[ir]=1/r[ir]
	return psi
class InletNormalVelocity():
	def __init__(self, S): self.S = S

	def __call__(self, x): return self.S*grabovski_berger(x[1])
class yaj():
	def __init__(self,meshpath,dnspath,m,Re,S,n_S):
		# Newton solver
		self.nu	 =1. #viscosity prefator
		self.n_nu=1 #number of visocity iterations
		self.S   =S #swirl amplitude relative to main flow
		self.n_S =n_S #number of swirl iterations
		self.m  =m # azimuthal decomposition

		# Fundamental flow type
		self.dnspath=dnspath
		self.private_path  	='doing/'
		self.resolvent_path	='resolvent/'
		self.eig_path		='eigenvalues/'
		#Re_string		='_Re='+str(Re)+'_'
		self.baseflow_string='_S='+f"{S:00.3f}"
		self.save_string   	=self.baseflow_string+'_m='+str(m)
		
		# Mesh from file
		with XDMFFile(COMM_WORLD, meshpath, "r") as file:
			self.mesh = file.read_mesh(name="Grid")
		
		# Taylor Hodd elements ; stable element pair
		FE_vector=ufl.VectorElement("Lagrange",self.mesh.ufl_cell(),2,3)
		FE_scalar=ufl.FiniteElement("Lagrange",self.mesh.ufl_cell(),1)
		self.Space=dolfinx.FunctionSpace(self.mesh,ufl.MixedElement([FE_vector,FE_scalar]))	# full vector function space
		# Subspaces
		self.U_space =  self.Space.sub(0).collapse()
		self.u_space =self.U_space.sub(0).collapse()

		# 3 components zero
		self.Zeros=dolfinx.Function(self.U_space)
		with self.Zeros.vector.localForm() as zero_loc: zero_loc.set(0)
		# One component zero
		self.zeros=dolfinx.Function(self.u_space)
		with self.zeros.vector.localForm() as zero_loc: zero_loc.set(0)
		# One component one
		self.ones=dolfinx.Function(self.u_space)
		with self.ones.vector.localForm()  as one_loc:  one_loc.set(1)

		# Test functions
		testFunction = ufl.TestFunction(self.Space)
		self.Test = [ufl.as_vector((testFunction[0],testFunction[1],testFunction[2])),testFunction[3]]
		self.Trial = ufl.TrialFunction(self.Space)

		self.q = dolfinx.Function(self.Space) # initialisation of q
		self.HotStart()
		self.r = dolfinx.Function(self.u_space)
		self.r.interpolate(lambda x: x[1])
		self.SpongedReynolds(Re)
		
	# Sponged Reynolds number
	def SpongedReynolds(self,Re):
		Re_s=.1
		self.Re=dolfinx.Function(dolfinx.FunctionSpace(self.mesh,ufl.FiniteElement("Lagrange",self.mesh.ufl_cell(),2)))
		def csi(a,b): return .5*(1+np.tanh(4*np.tan(-np.pi/2+np.pi*np.abs(a-b)/l)))
		def Ref(x):
			Rem=np.ones(x[0].size)*Re
			x_ext=x[0]>x_max
			Rem[x_ext]=Re		 +(Re_s-Re) 	   *csi(np.minimum(x[0][x_ext],x_max+l),x_max)
			r_ext=x[1]>r_max
			Rem[r_ext]=Rem[r_ext]+(Re_s-Rem[r_ext])*csi(np.minimum(x[1][r_ext],r_max+l),r_max)
			return Rem
		self.Re.interpolate(Ref)
		
	# Memoisation routine - find closest in Re and S
	def HotStart(self):
		file_names = [f for f in os.listdir(self.dnspath+self.private_path+'dat/') if f[-3:]=="dat"]
		closest_file_name=self.dnspath+"last_baseflow.dat"
		d=np.infty
		for file_name in file_names:
			Sd =float(file_name[11:16]) # Take advantage of file format 
			fd=abs(self.S-Sd)#+abs(Re-Red)
			if fd<d: d,closest_file_name=fd,self.dnspath+self.private_path+'dat/'+file_name
		
		viewer = PETSc.Viewer().createMPIIO(closest_file_name,'r',COMM_WORLD)
		self.q.vector.load(viewer)
		self.q.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
		
	def FreeSlip(self):
		# Compute DoFs
		dofs_misc_r  = dolfinx.fem.locate_dofs_geometrical((self.Space.sub(0).sub(1), self.u_space), misc)
		dofs_misc_th = dolfinx.fem.locate_dofs_geometrical((self.Space.sub(0).sub(2), self.u_space), misc)

		# Actual BCs
		bcs_misc_r  = dolfinx.DirichletBC(self.zeros, dofs_misc_r,  self.Space.sub(0).sub(1)) # u_r=0 at r=R
		bcs_misc_th = dolfinx.DirichletBC(self.zeros, dofs_misc_th, self.Space.sub(0).sub(2)) # u_th=0 at r=R

		return [bcs_misc_r,bcs_misc_th]

	def AxisSymmetry(self):
		# Compute DoFs
		dofs_symmetry_r  = dolfinx.fem.locate_dofs_geometrical((self.Space.sub(0).sub(1), self.u_space), symmetry)
		dofs_symmetry_th = dolfinx.fem.locate_dofs_geometrical((self.Space.sub(0).sub(2), self.u_space), symmetry)

		# Actual BCs
		bcs_symmetry_r  = dolfinx.DirichletBC(self.zeros, dofs_symmetry_r,  self.Space.sub(0).sub(1)) # u_r=0 at r=0
		bcs_symmetry_th = dolfinx.DirichletBC(self.zeros, dofs_symmetry_th, self.Space.sub(0).sub(2)) # u_th=0 at r=0

		return [bcs_symmetry_r,bcs_symmetry_th]

	# Baseflow
	def BoundaryConditions(self,S):
		# Modified vortex that goes to zero at top boundary
		u_inlet_th=dolfinx.Function(self.u_space)
		self.inlet_normal_velocity=InletNormalVelocity(S)
		u_inlet_th.interpolate(self.inlet_normal_velocity)
		u_inlet_th.x.scatter_forward()

		# Compute DoFs
		dofs_inlet_x  = dolfinx.fem.locate_dofs_geometrical((self.Space.sub(0).sub(0), self.u_space), inlet)
		dofs_inlet_r  = dolfinx.fem.locate_dofs_geometrical((self.Space.sub(0).sub(1), self.u_space), inlet)
		dofs_inlet_th = dolfinx.fem.locate_dofs_geometrical((self.Space.sub(0).sub(2), self.u_space), inlet)

		# Actual BCs
		bcs_inlet_x  = dolfinx.DirichletBC(self.ones,  dofs_inlet_x,  self.Space.sub(0).sub(0)) # u_x=1
		bcs_inlet_r  = dolfinx.DirichletBC(self.zeros, dofs_inlet_r,  self.Space.sub(0).sub(1)) # u_r=0
		bcs_inlet_th = dolfinx.DirichletBC(u_inlet_th, dofs_inlet_th, self.Space.sub(0).sub(2)) # u_th=S*psi(r) at x=0
																						    	# x=X entirely handled by implicit Neumann
		self.bcs = [bcs_inlet_x, bcs_inlet_r, bcs_inlet_th]
		self.bcs.extend(self.FreeSlip())
		self.bcs.extend(self.AxisSymmetry())

	def BoundaryConditionsPerturbations(self):
		# Compute DoFs
		dofs_inlet_r  	 = dolfinx.fem.locate_dofs_geometrical((self.Space.sub(0).sub(1), self.u_space), inlet)
		dofs_symmetry    = dolfinx.fem.locate_dofs_geometrical((self.Space.sub(0), 		  self.u_space), symmetry)

		# Actual BCs
		bcs_inlet_r		= dolfinx.DirichletBC(self.zeros, dofs_inlet_r,  self.Space.sub(0).sub(1)) # u_r=0 at x=0
		bcs_symmetry_r  = dolfinx.DirichletBC(self.Zeros, dofs_symmetry, self.Space.sub(0)) 	# u_r=0 at r=0
		
		self.bcps = [bcs_inlet_r]
		self.bcps.extend(self.FreeSlip())
		if self.m==0: self.bcp.extend(self.AxisSymmetry())
		else:		  self.bcp.append(bcs_symmetry_r)

	def ComputeIndices(self):
		# Collect all dirichlet boundary dof indices
		bcinds = []
		for b in self.bcp:
			bcdict = b.get_boundary_values()
			bcinds.extend(bcdict.keys())

		# total number of dofs
		N = self.Space.dim()

		# indices of free nodes
		self.freeinds = np.setdiff1d(range(N),bcinds,assume_unique=True).astype(np.int32)

	def NonlinearOperator(self,m):
		u,p=extract_up(self.q)
		r,test_u,test_m=self.r,self.Test[0],self.Test[1]
		
		#mass (variational formulation)
		F = ufl.inner(div(u,r,m),test_m)*r*ufl.dx
		#momentum (different test functions and IBP)
		F += 		 ufl.inner(grad(u,r,m)*u,  	 test_u)     *r*ufl.dx # Convection
		F += self.mu*ufl.inner(grad(u,r,m), grad(test_u,r,m))*r*ufl.dx # Diffusion
		F -= 		 ufl.inner(p, 			 div(test_u,r,m))*r*ufl.dx # Pressure
		return F

	def Newton(self):
		if self.n_S>1: Ss= np.cos(np.pi*np.linspace(self.n_S,0,self.n_S)/2/self.n_S)*self.S # Chebychev spacing
		else: 		   Ss=[self.S]
		self.BoundaryConditions(Ss[0]) #for temporal-dependant boundary condition
		for S_current in Ss: 	# Increase swirl
			for nu_current in np.linspace(self.nu,1,self.n_nu): # Decrease viscosity (non physical but helps CV)
				print("viscosity prefactor: ", nu_current)
				print("swirl intensity: ",	    S_current)
				self.mu=nu_current/self.Re #recalculate viscosity with prefactor
				self.inlet_normal_velocity.S=S_current
				# Compute form
				base_form = self.NonlinearOperator(0) #no azimuthal decomposition for base flow (so no imaginary part to operator)
				dbase_form = ufl.derivative(base_form,self.q,self.Trial)
				problem = dolfinx.fem.NonlinearProblem(base_form,self.q,bcs=self.bcs, J=dbase_form)
				solver  = dolfinx.NewtonSolver(COMM_WORLD, problem)
				solver.rtol=eps
				solver.relaxation_parameter=rp
				solver.max_iter=30
				solver.atol=ae   
				solver.solve(self.q)
				if nu_current==1:
					#write results in private_path for a given mu
					u,p = self.q.split()
					with VTKFile(COMM_WORLD, self.dnspath+self.private_path+"print/u_S="+f"{S_current:00.3f}"+".pvd","w") as vtk:
						vtk.write([u._cpp_object])
					viewer = PETSc.Viewer().createMPIIO(self.dnspath+self.private_path+"dat/baseflow_S="+f"{S_current:00.3f}"+".dat", 'w', COMM_WORLD)
					self.q.vector.view(viewer)
					print(".pvd, .dat written!")
				
		#write result of current mu
		with VTKFile( COMM_WORLD, self.dnspath+"last_u.pvd","w") as vtk:
			vtk.write([u._cpp_object])
		viewer = PETSc.Viewer().createMPIIO(self.dnspath+"last_baseflow.dat", 'w', COMM_WORLD)
		self.q.vector.view(viewer)
		print(self.dnspath+"last_baseflow.dat written!")

	def ComputeAM(self):
		# Go complex
		#matrix A (m*m): Jacobian calculated by hand
		form=self.NonlinearOperator(self.m)
		dform=ufl.derivative(form,self.q,self.Trial)
		self.BoundaryConditionsPerturbations()
		Aa = dolfinx.fem.assemble_matrix(dform, bcs=self.bcps)
		Aa.assemble().tocsc()
		set_trace()

		#forcing norm M (m*m): here we choose ux^2+ur^2+uth^2 as forcing norm
		#other userdefined norm can be used, to be added later
		up_r = ufl.as_vector((self.Trial[0],self.Trial[1],self.Trial[2]))
		#up_i = as_vector((	 Trial_i[0],   Trial_i[1],   Trial_i[2]))
		M_form=ufl.dot(up_r,self.Test[0])*self.r*ufl.dx
		#M_form=dot(up_r,self.Test[0])*r*dx+dot(up_i,self.Test[0])*self.r*dx
		Ma = dolfinx.fem.assemble_matrix(M_form)
		Ma.assemble().tocsc()
		#self.M = Ma[self.freeinds,:][:,self.freeinds]

	def Getw0(self):
		U,p=self.q.split()
		u,v,w=U.split()
		return np.min(u.compute_point_values())

	def Resolvent(self,k,freq_list):
		print("check base flow max and min in u:",np.max(self.q.vector()[:]),",",np.min(self.q.vector()[:]))

		#matrix B (m*m): with matrix A form altogether the resolvent operator
		up=ufl.as_vector((self.Trial[0],self.Trial[1],self.Trial[2]))
		pp=self.Trial[3]
		B_form=ufl.dot(up,self.Test[0])*self.r*ufl.dx
		Ba = dolfinx.fem.assemble_matrix(B_form)
		Ba.assemble().tocsc()

		#response norm Mr (m*m): here we choose the same as forcing norm
		Mr, Mf = self.M, self.M

		#quadrature Q (m*m): it is required to compensate the quadrature in resolvent operator R, because R=(A-i*omegaB)^(-1)
		Q_form=ufl.dot(up,self.Test[0])*self.r*ufl.dx+pp*self.Test[1]*self.r*ufl.dx
		Qa = dolfinx.fem.assemble_matrix(Q_form)
		Qa.assemble().tocsc()

		#matrix P (m*n) reshapes forcing vector (n*1) to (m*1). In principal, it contains only 0 and 1 elements.
		#It can also restrict the flow regions of forcing, to be implemented later. 
		#A note for very rare case: if one wants to damp but not entirely eliminate the forcing in some regions (not the case here), one can put values between 0 and 1. In that case, the matrix I in the following is not the same as P. 
		
		index_forcing=self.Space.sub(0).dofmap().dofs() #get all index related to u
		index_forcing.sort()

		row_ind=np.intersect1d(self.freeinds,index_forcing) #get free index related to u
		row_ind.sort()
		m=len(self.Space.dofmap().dofs())
		n=len(row_ind)
		col_ind=np.arange(n)
		Pa=sps.csc_matrix((np.ones(n),(row_ind,col_ind)),(m,n))

		#matrix I (m*n) reshapes forcing matrix Mf (m*m) to I^T*Mf*I (n*n). The matrix I can be different from P in that very rare case remarked above.
		Ia=Pa

		B  = Ba[self.freeinds,:][:,self.freeinds]
		P  = Pa[self.freeinds]
		I  = Ia[self.freeinds]
		Q  = Qa[self.freeinds,:][:,self.freeinds]

		Q_shape=np.shape(Q)
		print('matrix Q size: '+str(Q_shape))
		P_shape=np.shape(P)
		print('matrix P size: '+str(P_shape))

		for freq in freq_list:
			R = la.splu(-self.A-2*np.pi*1j*freq*B,permc_spec=3)
			# get response linear operator P^H*Q^H*R^H*Mr*R*Q*P
			def lhs(f): return P.transpose()*Q.transpose()*R.solve(Mr*R.solve(Q*P*f),trans='H')

			LHS = la.LinearOperator((min(P_shape),min(P_shape)),matvec=lhs,dtype='complex')

			# forcing linear operator is on the rhs M=I.transpose()*Mf*I
			gains,eigenvectors = la.eigs(LHS, k=k, M=I.transpose()*Mf*I, sigma=None,  maxiter=100, tol=ae, return_eigenvectors=True)
			
			#write forcing and response
			f=eigenvectors
			r=R.solve(Q*P*f)

			ua = dolfinx.Function(self.Space) #declaration for efficiency

			for i in range(k):
				ua.vector()[self.freeinds] = np.abs(P*f[:,i])
				u,p  = ua.split()
				with VTKFile(COMM_WORLD, self.dnspath+self.resolvent_path+"forcing_u" +self.save_string+"f="+f"{freq:00.3f}"+"_n="+f"{i+1:1d}"+".pvd","w") as vtk:
					vtk.write([u._cpp_object])
				ua.vector()[self.freeinds] = np.abs(r[:,i])
				u,p  = ua.split()
				with VTKFile(COMM_WORLD, self.dnspath+self.resolvent_path+"response_u"+self.save_string+"f="+f"{freq:00.3f}"+"_n="+f"{i+1:1d}"+".pvd","w") as vtk:
					vtk.write([u._cpp_object])
			
			#write gains
			np.savetxt(self.dnspath+self.resolvent_path+"gains"+self.save_string+"f="+f"{freq:00.3f}"+".dat",np.real(gains))

	def Eigenvalues(self,sigma,k,flag_mode,savematt,loadmatt):
		print("check base flow max and min in u:",np.max(self.q.vector()[:]),",",np.min(self.q.vector()[:]))

		#RHS
		if flag_mode==0:
			print("save matrix to file "+savematt+self.save_string+".mat and quit!")
			from scipy.io import savemat
			mdic = {"A": self.A, "M": self.M}
			savemat(savematt+self.save_string+".mat", mdic)
			return 0
		elif flag_mode==1:
			print("load matlab result from file "+loadmatt)
			from scipy.io import loadmat
			mdic=loadmat(loadmatt)
			vecs=mdic['V'] #if KeyError: 'V', it means the eigenvalue results are not saved into .mat
			vals=np.diag(mdic['D'])
		elif flag_mode==2:			
			print("Computing eigenvalues/vectors in Python!")
			ncv = max(10,2*k)
			vals, vecs = la.eigs(self.A, k=k, M=self.M, sigma=sigma, maxiter=60, tol=ae, ncv=ncv)
		else:
			print("Operation mode for eigenvalues is not correct. Nothing done.")
			return 0

		# only writing real parts of eigenvectors to file
		ua = dolfinx.Function(self.Space)
		flag_video=0 #1: export animation
		for i in range(0,k+1,k//10+1):
			ua.vector()[self.freeinds] = vecs[:,i].real

			u,p  = ua.split()
			with VTKFile(COMM_WORLD, self.dnspath+self.eig_path+"u/evec_u"+self.save_string+"_n="+str(i+1)+".pvd","w") as vtk:
				vtk.write([u._cpp_object])
			with VTKFile(COMM_WORLD, self.dnspath+self.eig_path+"u/evec_u"+self.save_string+"_n="+str(i+1)+".pvd","w") as vtk:
				vtk.write([p._cpp_object])
			if flag_video: # export animation
				print("Exporting video for eig "+str(i+1))
				angSteps = 20
				angList = list(2*np.pi/angSteps*np.arange(0,angSteps+1))

				angle0=np.angle(vecs[:,i])
				abs0=np.absolute(vecs[:,i])
				for k in range(0,angSteps+1):
					angle = angList[k]-angle0
					amp = abs0*np.cos(angle)
					ua.vector()[self.freeinds] = amp
					if self.label=='lowMach':
						u,p,rho  = ua.split()
					if self.label=='lowMach_reacting':
						u,p,rho,y  = ua.split()
					File(self.dnspath+self.eig_path+"anim_rho_nu="+f"{self.nu:00.3f}"+"_S="+f"{self.S:00.3f}"+"_"+str(i+1)+"_"+str(k)+".pvd") << rho
					File(self.dnspath+self.eig_path+"anim_u_nu="+f"{self.nu:00.3f}"+"_S="+f"{self.S:00.3f}"+"_"+str(i+1)+"_"+str(k)+".pvd") << u
					if self.label=='lowMach_reacting':
						File(self.dnspath+self.eig_path+"anim_y_nu="+f"{self.nu:00.3f}"+"_S="+f"{self.S:00.3f}"+"_"+str(i+1)+"_"+str(k)+".pvd") << y
		#write eigenvalues
		np.savetxt(self.dnspath+self.eig_path+"evals"+self.save_string+".dat",np.column_stack([vals.real, vals.imag]))