# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
from fileinput import filename
import re
import numpy as np
import dolfinx as dfx
import os, ufl, shutil
from dolfinx.io import XDMFFile
from petsc4py import PETSc as pet
from mpi4py.MPI import COMM_WORLD as comm

p0=comm.rank==0

# Simple handler
def meshConvert(path:str,cell_type:str) -> None:
	if p0:
		import meshio #pip3 install --no-binary=h5py h5py meshio
		gmsh_mesh = meshio.read(path+".msh")
		# Write it out again
		ps = gmsh_mesh.points[:,:2]
		cs = gmsh_mesh.get_cells_type(cell_type)
		dolfinx_mesh = meshio.Mesh(points=ps, cells={cell_type: cs})
		meshio.write(path+".xdmf", dolfinx_mesh)

# Swirling Parallel Yaj
class SPY:
	def __init__(self, params:dict, datapath:str, Ref, nutf, direction_map:dict) -> None:
		# Direction dependant
		self.direction_map=direction_map

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
		with XDMFFile(comm, meshpath, "r") as file:
			self.mesh = file.read_mesh(name="Grid")
		# Extraction of r
		self.r = ufl.SpatialCoordinate(self.mesh)[1]
	
		# Finite elements & function spaces
		FE_vector=ufl.VectorElement("Lagrange",self.mesh.ufl_cell(),2,3)
		FE_scalar=ufl.FiniteElement("Lagrange",self.mesh.ufl_cell(),1)
		V = dfx.FunctionSpace(self.mesh,FE_scalar)
		self.TH=dfx.FunctionSpace(self.mesh,FE_vector*FE_scalar)	# Taylor Hodd elements ; stable element pair
		self.u_space, self.u_dofs = self.TH.sub(0).collapse(collapsed_dofs=True)
		self.u_dofs=np.array(self.u_dofs)
		
		# Test & trial functions
		self.trial = ufl.TrialFunction(self.TH)
		self.test  = ufl.TestFunction( self.TH)
		
		# Re computation
		self.Re = Ref(self)
		self.q = dfx.Function(self.TH) # Initialisation of q
		self.nut = dfx.Function(V)
		self.nutf = lambda S: nutf(self,S)

		# BCs essentials
		self.dofs = np.empty(0,dtype=np.int32)
		self.bcs=[]

	# Jet geometry
	def symmetry(self, x:ufl.SpatialCoordinate) -> np.ndarray:
		return np.isclose(x[self.direction_map['r']],0,self.params['atol']) # Axis of symmetry at r=0

	# Gradient with x[0] is x, x[1] is r, x[2] is theta
	def rgrad(self,v,m):
		return ufl.as_tensor([[self.r*v[0].dx(self.direction_map['x']), self.r*v[0].dx(self.direction_map['r']), m*1j*v[0]],
							  [self.r*v[1].dx(self.direction_map['x']), self.r*v[1].dx(self.direction_map['r']), m*1j*v[1]-v[2]],
							  [self.r*v[2].dx(self.direction_map['x']), self.r*v[2].dx(self.direction_map['r']), m*1j*v[2]+v[1]]])

	def gradr(self,v,m):
		return ufl.as_tensor([[self.r*v[0].dx(self.direction_map['x']), v[0]+self.r*v[0].dx(self.direction_map['r']), m*1j*v[0]],
					  		  [self.r*v[1].dx(self.direction_map['x']), v[1]+self.r*v[1].dx(self.direction_map['r']), m*1j*v[1]-v[2]],
							  [self.r*v[2].dx(self.direction_map['x']), v[2]+self.r*v[2].dx(self.direction_map['r']), m*1j*v[2]+v[1]]])

	# Same for divergent
	def rdiv(self,v,m): return self.r*v[0].dx(self.direction_map['x']) +   v[1] + self.r*v[1].dx(self.direction_map['r']) + m*1j*v[2]

	def divr(self,v,m): return self.r*v[0].dx(self.direction_map['x']) + 2*v[1] + self.r*v[1].dx(self.direction_map['r']) + m*1j*v[2]

	# Heart of this entire code
	def navierStokes(self) -> ufl.Form:
		# Shortforms
		r=self.r
		rdiv,divr,rgrad,gradr=self.rdiv,self.divr,self.rgrad,self.gradr
		u,p=ufl.split(self.q)
		v,s=ufl.split(self.test)
		
		# Mass (variational formulation)
		F  = ufl.inner( rdiv(u,0),	  r*s)
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
		F  = ufl.inner( rdiv(u, m),	   r*s)
		# Momentum (different test functions and IBP)
		F += ufl.inner(rgrad(ub,0)*u,  r*v)    		 # Convection
		F += ufl.inner(rgrad(u, m)*ub, r*v)
		F += ufl.inner(rgrad(u, m),gradr(v,m))*(1/self.Re+self.nut) # Diffusion
		F -= ufl.inner(r*p,			divr(v,m)) 		 # Pressure
		return F*ufl.dx
		
	# Code factorisation
	def constantBC(self, direction:chr, boundary, value=0) -> tuple:
		sub_space=self.TH.sub(0).sub(self.direction_map[direction])
		sub_space_collapsed=sub_space.collapse()
		# Compute proper values
		constant=dfx.Function(sub_space_collapsed)
		with constant.vector.localForm() as zero_loc: zero_loc.set(value)
		# Compute unflattened DoFs (don't care for flattened ones)
		dofs = dfx.fem.locate_dofs_geometrical((sub_space, sub_space_collapsed), boundary)
		# Actual BCs
		bcs = dfx.DirichletBC(constant, dofs, sub_space) # u_i=value at boundary
		return dofs,bcs

	# Encapsulation	
	def applyBCs(self, dofs:np.ndarray,bcs:list):
		self.dofs=np.union1d(dofs,self.dofs)
		self.bcs.extend(bcs)

	def applyHomogeneousBCs(self, tup:list):
		for marker,directions in tup:
			for direction in directions:
				dofs,bcs=self.constantBC(direction,marker)
				self.applyBCs(dofs,[bcs])
	
	# Memoisation routine - find closest in param
	def loadStuff(self,param,path,pattern,vector) -> None:
		closest_file_name=path
		if not os.path.isdir(path):
			comm.barrier() # Wait for all other processors
			if p0: os.mkdir(path)
			return
		file_names = [f for f in os.listdir(path) if f[-3:]=="dat"]
		d=np.infty
		for file_name in file_names:
			match = re.search(pattern,file_name)
			param_file = float(match.group(1)) # Take advantage of file format
			match = re.search(r'_n=([0-9]*)',file_name)
			n = int(match.group(1))
			if n!=comm.size: continue # Don't read if != proc nb
			fd = abs(param-param_file)
			if fd<d: d,closest_file_name=fd,path+file_name
		viewer = pet.Viewer().createMPIIO(closest_file_name, 'r', comm)
		vector.load(viewer)
		vector.ghostUpdate(addv=pet.InsertMode.INSERT, mode=pet.ScatterMode.FORWARD)
		# Loading eddy viscosity too
		if comm.rank==0:
			print("Loaded "+closest_file_name+" as part of memoisation scheme")

	def saveStuff(self,dir:str,name:str,fun:dfx.Function) -> None:
		if p0 and not os.path.isdir(dir): os.mkdir(dir)
		with XDMFFile(comm, dir+name+".xdmf", "w") as xdmf:
			xdmf.write_mesh(self.mesh)
			xdmf.write_function(fun)
	
	def saveStuffMPI(self,dir:str,name:str,vec:pet.Vec) -> None:
		if p0 and not os.path.isdir(dir): os.mkdir(dir)
		viewer = pet.Viewer().createMPIIO(dir+name+f"_n={comm.size:d}.dat", 'w', comm)
		vec.view(viewer)

	# Converters
	def datToNpy(self,fi,fo) -> None:
		viewer = pet.Viewer().createMPIIO(fi, 'r', comm)
		self.q.vector.load(viewer)
		self.q.x.scatter_forward()
		np.save(fo,self.q.x.array)

	def datToNpyAll(self) -> None:
		file_names = [f for f in os.listdir(self.dat_real_path) if f[-3:]=="dat"]
		if p0 and not os.path.isdir(self.npy_path): os.mkdir(self.npy_path)
		for file_name in file_names:
			self.datToNpy(self.dat_real_path+file_name,
						  self.npy_path+file_name[:-3]+'npy')
		shutil.rmtree(self.dat_real_path)

	def npyToDat(self,fi,fo) -> None:
		self.q.vector.array.real=np.load(fi,allow_pickle=True)
		self.q.x.scatter_forward()
		viewer = pet.Viewer().createMPIIO(fo, 'w', comm)
		self.q.vector.view(viewer)
	
	def npyToDatAll(self) -> None:
		file_names = [f for f in os.listdir(self.npy_path)]
		if p0 and not os.path.isdir(self.dat_complex_path): os.mkdir(self.dat_complex_path)
		for file_name in file_names:
			self.npyToDat(self.npy_path+file_name,
						  self.dat_complex_path+file_name[:-3]+'dat')
		shutil.rmtree(self.npy_path)
	
	# Quick check functions
	def sanityCheckU(self):
		u,_=self.q.split()
		self.saveStuff("","sanity_check_u",u)

	def sanityCheckBCs(self):
		v=dfx.Function(self.TH)
		v.vector.zeroEntries()
		v.x.array[self.dofs]=np.ones(self.dofs.size)
		u,_=v.split()
		self.saveStuff("","sanity_check_bcs",u)