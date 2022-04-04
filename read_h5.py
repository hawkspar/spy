import ufl, h5py
import numpy as np
from scipy.interpolate import interp2d
import dolfinx as dfx
from dolfinx.io import XDMFFile
from mpi4py.MPI import COMM_WORLD

case="nozzle"
mesh_path="../cases/"+case+"/"+case+"_2D.xdmf"
with XDMFFile(COMM_WORLD, mesh_path, "r") as file:
    mesh = file.read_mesh(name="Grid")

with XDMFFile(COMM_WORLD, "front.xdmf", "r") as file:
    mesh = file.read_mesh(name="Grid")

# Taylor Hodd elements ; stable element pair
FE_vector=ufl.VectorElement("Lagrange",mesh.ufl_cell(),2,3)
FE_scalar=ufl.FiniteElement("Lagrange",mesh.ufl_cell(),1)
Space=dfx.FunctionSpace(mesh,FE_scalar)
p = dfx.Function(Space)
#Space=dfx.FunctionSpace(mesh,FE_vector)
#Ux = dfx.Function(Space)

with h5py.File('front.h5', 'r') as ifile:
    x_dat = ifile['Data4'][()]
    p_dat = ifile['Data6'][()]
    #U_dat = ifile['Data7'][()]
    print(p.vector.getSize())
    print(p_dat)
    p.vector[:] = p_dat

#inter=interp2d(x_dat[:,0],x_dat[:,1],U_dat[:,0])

#Ux.interpolate(lambda x: inter(x[0],x[1]))

with XDMFFile(COMM_WORLD, "sanity_check.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(p)