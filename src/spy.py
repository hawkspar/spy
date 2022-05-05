# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import numpy as np
import dolfinx as dfx
import os, ufl, shutil
from dolfinx.io import XDMFFile
from petsc4py import PETSc as pet
from mpi4py.MPI import COMM_WORLD

# Swirling Parallel Yaj
class SPY:
	def __init__(self, params:dict, datapath:str, Ref, nutf) -> None:
		# TBI : problem dependent
		self.direction_map={'x':0,'r':1,'th':2}

		# Solver parameters (Newton mostly, but also eig)
		self.params=params

		# Paths
		self.case_path		 ='/home/shared/cases/'+datapath
		self.baseflow_path   =self.case_path+'baseflow/'
		self.nut_path		 =self.baseflow_path+'nut/'
		self.dat_real_path	 =self.baseflow_path+'dat_real/'
		self.dat_complex_path=self.baseflow_path+'dat_complex/'
		self.print_path		 =self.baseflow_path+'print/'
		self.npy_path		 =self.baseflow_path+'npy/'
		self.resolvent_path	 =self.case_path+'resolvent/'
		self.eig_path		 =self.case_path+'eigenvalues/'

		# Mesh from file
		meshpath=self.case_path+datapath[:-1]+".xdmf"
		with XDMFFile(COMM_WORLD, meshpath, "r") as file:
			self.mesh = file.read_mesh(name="Grid")
		
		# Taylor Hodd elements ; stable element pair
		FE_vector=ufl.VectorElement("Lagrange",self.mesh.ufl_cell(),2,3)
		FE_scalar=ufl.FiniteElement("Lagrange",self.mesh.ufl_cell(),1)
		self.TH=dfx.FunctionSpace(self.mesh,FE_vector*FE_scalar)	# full vector function space
		V = dfx.FunctionSpace(self.mesh,FE_scalar)
		
		# test & trial functions
		self.test  = ufl.TestFunction(self.TH)
		self.trial = ufl.TrialFunction(self.TH)
		
		# Extraction of r and Re computation
		self.r = ufl.SpatialCoordinate(self.mesh)[1]
		self.Re=dfx.Function(V)
		self.Re.interpolate(Ref)
		self.q = dfx.Function(self.TH) # Initialisation of q
		self.nut = dfx.Function(V)
		self.nutf = lambda S: nutf(self,S)

	# Jet geometry
	def symmetry(self, x:ufl.SpatialCoordinate) -> np.ndarray:
		return np.isclose(x[1],0,self.params['atol']) # Axis of symmetry at r=0

	# Gradient with x[0] is x, x[1] is r, x[2] is theta
	def rgrad(self,v,m):
		return ufl.as_tensor([[self.r*v[0].dx(0), self.r*v[0].dx(1), m*1j*v[0]],
							  [self.r*v[1].dx(0), self.r*v[1].dx(1), m*1j*v[1]-v[2]],
							  [self.r*v[2].dx(0), self.r*v[2].dx(1), m*1j*v[2]+v[1]]])

	def gradr(self,v,m):
		return ufl.as_tensor([[self.r*v[0].dx(0), v[0]+self.r*v[0].dx(1), m*1j*v[0]],
					  		  [self.r*v[1].dx(0), v[1]+self.r*v[1].dx(1), m*1j*v[1]-v[2]],
							  [self.r*v[2].dx(0), v[2]+self.r*v[2].dx(1), m*1j*v[2]+v[1]]])

	# Same for divergent
	def rdiv(self,v,m): return self.r*v[0].dx(0) +   v[1] + self.r*v[1].dx(1) + m*1j*v[2]

	def divr(self,v,m): return self.r*v[0].dx(0) + 2*v[1] + self.r*v[1].dx(1) + m*1j*v[2]

	# Heart of this entire code
	def navierStokes(self) -> ufl.Form:
		# Shortforms
		r=self.r
		rdiv,divr,rgrad,gradr=self.rdiv,self.divr,self.rgrad,self.gradr
		u,p=ufl.split(self.q)
		v,s=ufl.split(self.test)
		
		# Mass (variational formulation)
		F  = ufl.inner( rdiv(u,0),		s)
		# Momentum (different test functions and IBP)
		F += ufl.inner(rgrad(u,0)*u,  r*v)       	   # Convection
		F += ufl.inner(rgrad(u,0),gradr(v,0))*(1/self.Re+self.nut) # Diffusion
		F -= ufl.inner(r*p,		   divr(v,0)) 	  	   # Pressure
		return F*ufl.dx
		
	# Not automatic because of convection term
	def linearisedNavierStokes(self,m:int) -> ufl.Form:
		# Shortforms
		r=self.r
		rdiv,divr,rgrad,gradr=self.rdiv,self.divr,self.rgrad,self.gradr
		u, p=ufl.split(self.trial)
		ub,_=ufl.split(self.q) # Baseflow
		v, s=ufl.split(self.test)
		
		# Mass (variational formulation)
		F  = ufl.inner( rdiv(u, m), 	 s)
		# Momentum (different test functions and IBP)
		F += ufl.inner(rgrad(ub,0)*u,  r*v)    		 # Convection
		F += ufl.inner(rgrad(u, m)*ub, r*v)
		F += ufl.inner(rgrad(u, m),gradr(v, m))*(1/self.Re+self.nut) # Diffusion
		F -= ufl.inner(r*p,			 divr(v,m)) 		 # Pressure
		return F*ufl.dx
		
	# Code factorisation
	def constantBC(self, direction:chr, boundary, value=0) -> tuple:
		sub_space=self.TH.sub(0).sub(self.direction_map[direction])
		sub_space_collapsed=sub_space.collapse()
		# Compute proper zeros
		constant=dfx.Function(sub_space_collapsed)
		with constant.vector.localForm() as zero_loc: zero_loc.set(value)
		# Compute DoFs
		dofs = dfx.fem.locate_dofs_geometrical((sub_space, sub_space_collapsed), boundary)
		# Actual BCs
		bcs = dfx.DirichletBC(constant, dofs, sub_space) # u_i=value at boundary
		return dofs[0], bcs # Only return unflattened dofs
	
	# Memoisation routine - find closest in S
	def loadStuff(self,S,last_name,path,offset,vector) -> None:
		closest_file_name=path+last_name
		if not os.path.isdir(path): os.mkdir(path)
		file_names = [f for f in os.listdir(path) if f[-3:]=="dat"]
		d=np.infty
		for file_name in file_names:
			Sd = float(file_name[offset:offset+5]) # Take advantage of file format 
			fd = abs(S-Sd)
			if fd<d: d,closest_file_name=fd,path+file_name
		viewer = pet.Viewer().createMPIIO(closest_file_name, 'r', COMM_WORLD)
		vector.load(viewer)
		vector.ghostUpdate(addv=pet.InsertMode.INSERT, mode=pet.ScatterMode.FORWARD)
		# Loading eddy viscosity too
		if COMM_WORLD.rank==0:
			print("Loaded "+closest_file_name+" as part of memoisation scheme")

	# Converters
	def datToNpy(self,fi,fo) -> None:
		viewer = pet.Viewer().createMPIIO(fi, 'r', COMM_WORLD)
		self.q.vector.load(viewer)
		self.q.vector.ghostUpdate(addv=pet.InsertMode.INSERT, mode=pet.ScatterMode.FORWARD)
		np.save(fo,self.q.x.array)

	def datToNpyAll(self) -> None:
		file_names = [f for f in os.listdir(self.dat_real_path) if f[-3:]=="dat"]
		if not os.path.isdir(self.npy_path): os.mkdir(self.npy_path)
		for file_name in file_names:
			self.datToNpy(self.dat_real_path+file_name,
						  self.npy_path+file_name[:-3]+'npy')
		shutil.rmtree(self.dat_real_path)
		self.datToNpy(self.case_path+'last_baseflow_real.dat',self.case_path+'last_baseflow.npy')

	def npyToDat(self,fi,fo) -> None:
		self.q.vector.array.real=np.load(fi,allow_pickle=True)
		self.q.vector.ghostUpdate(addv=pet.InsertMode.INSERT, mode=pet.ScatterMode.FORWARD)
		viewer = pet.Viewer().createMPIIO(fo, 'w', COMM_WORLD)
		self.q.vector.view(viewer)
	
	def npyToDatAll(self) -> None:
		file_names = [f for f in os.listdir(self.npy_path)]
		if not os.path.isdir(self.dat_complex_path): os.mkdir(self.dat_complex_path)
		for file_name in file_names:
			self.npyToDat(self.npy_path+file_name,
						  self.dat_complex_path+file_name[:-3]+'dat')
		shutil.rmtree(self.npy_path)
		self.npyToDat(self.case_path+'last_baseflow.npy',self.case_path+'last_baseflow_complex.dat')
