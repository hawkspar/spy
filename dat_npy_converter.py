# coding: utf-8
"""
Created on Wed Oct 13 13:50:00 2021

@author: hawkspar
"""
import os, ufl
import numpy as np
import dolfinx as dfx
from dolfinx.io import XDMFFile
from petsc4py import PETSc as pet
from mpi4py.MPI import COMM_WORLD

class converter():
	def __init__(self,meshpath:str,datapath:str):
		# Paths
		self.datapath = datapath
		self.private_path  	='doing/'
		
		# Mesh from file
		with XDMFFile(COMM_WORLD, meshpath, "r") as file:
			self.mesh = file.read_mesh(name="Grid")
		
		# Taylor Hodd elements ; stable element pair
		FE_vector=ufl.VectorElement("Lagrange",self.mesh.ufl_cell(),2,3)
		FE_scalar=ufl.FiniteElement("Lagrange",self.mesh.ufl_cell(),1)
		self.Space=dfx.FunctionSpace(self.mesh,FE_vector*FE_scalar)	# full vector function space
		
		self.q = dfx.Function(self.Space) # Initialisation of q
		
	def datToNpy(self) -> None:
		file_names = [f for f in os.listdir(self.datapath+self.private_path+'dat/') if f[-3:]=="dat"]
		file_names.append(self.datapath+"last_baseflow.dat")
		for file_name in file_names:
			viewer = pet.Viewer().createMPIIO(file_name, 'r', COMM_WORLD)
			self.q.vector.load(viewer)
			self.q.vector.ghostUpdate(addv=pet.InsertMode.INSERT, mode=pet.ScatterMode.FORWARD)
			np.save(file_name[-3:]+'npy',self.q.x.array)
	
	def npyToDat(self) -> None:
		file_names = [f for f in os.listdir(self.datapath+self.private_path+'dat/') if f[-3:]=="npy"]
		file_names.append(self.datapath+"last_baseflow.npy")
		for file_name in file_names:
			self.q.x.array=np.load(file_name,allow_pickle=True)
			viewer = pet.Viewer().createMPIIO(file_name[-3:]+'dat', 'r', COMM_WORLD)
			self.q.vector.view(viewer)

converter("Mesh/validation/validation.xdmf","validation/").datToNpy()