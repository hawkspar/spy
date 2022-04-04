import meshio, ufl #pip3 install --no-binary=h5py h5py meshio
import numpy as np
import dolfinx as dfx
from dolfinx.io import XDMFFile
from mpi4py.MPI import COMM_WORLD
from petsc4py import PETSc as pet
from scipy.interpolate import interp2d

# Read mesh and point data
openfoam_mesh = meshio.read("../cases/nozzle/front3.xmf")
# Write it out again
cells = openfoam_mesh.get_cells_type("quad") 
dolfinx_fine_mesh = meshio.Mesh(points=openfoam_mesh.points[:,:2], cells={"quad": cells})
meshio.write("../cases/nozzle/nozzle_fine.xdmf", dolfinx_fine_mesh)
# Read it again in dolfinx
with XDMFFile(COMM_WORLD, "../cases/nozzle/nozzle_fine.xdmf", "r") as file:
    dolfinx_fine_mesh = file.read_mesh(name="Grid")
# Create FiniteElement, FunctionSpace & Functions
FE_vector_1=ufl.VectorElement("Lagrange",dolfinx_fine_mesh.ufl_cell(),1,3)
FE_vector_2=ufl.VectorElement("Lagrange",dolfinx_fine_mesh.ufl_cell(),2,3)
FE_scalar=ufl.FiniteElement(  "Lagrange",dolfinx_fine_mesh.ufl_cell(),1)
V_1=dfx.FunctionSpace(dolfinx_fine_mesh,FE_vector_1)
V_2=dfx.FunctionSpace(dolfinx_fine_mesh,FE_vector_2)
P=dfx.FunctionSpace(  dolfinx_fine_mesh,FE_scalar)
U_1 = dfx.Function(V_1)
U_2 = dfx.Function(V_2)
p = dfx.Function(P)
# Jiggle indexing
idcs = np.argsort(dolfinx_fine_mesh.geometry.input_global_indices).astype('int32')
vec_idcs = np.repeat(3*idcs,3)
vec_idcs[1::3]+=1
vec_idcs[2::3]+=2
# Map OpenFOAM data directy onto dolfinx vectors
U_1.vector[vec_idcs] = openfoam_mesh.point_data['U'].flatten()
p.vector[idcs] = openfoam_mesh.point_data['p']
# Fix orientation
e=np.pi/360
s,c=np.sin(e),np.cos(e)
Uy,Uz=U_1.vector[1::3],U_1.vector[2::3]
Uy,Uz=c*Uy+s*Uz,-s*Uy+c*Uz
U_1.vector[1::3],U_1.vector[2::3]=Uy,Uz
# Interpolation to higher order
U_2.interpolate(U_1)
"""
# Read smaller mesh in dolfinx
with XDMFFile(COMM_WORLD, "../cases/nozzle/nozzle_coarse.xdmf", "r") as file:
    dolfinx_coarse_mesh = file.read_mesh(name="Grid")
# Create coarser Functions
FE_vector=ufl.VectorElement("Lagrange",dolfinx_coarse_mesh.ufl_cell(),2,3)
FE_scalar=ufl.FiniteElement("Lagrange",dolfinx_coarse_mesh.ufl_cell(),1)
V=dfx.FunctionSpace(dolfinx_coarse_mesh,FE_vector)
P=dfx.FunctionSpace(dolfinx_coarse_mesh,FE_scalar)
U_o = dfx.Function(V)
p_o = dfx.Function(P)
# Interpolate results on coarser mesh
x_f,y_f=dolfinx_fine_mesh.geometry.x[:,0],  dolfinx_fine_mesh.geometry.x[:,1]
x_c,y_c=dolfinx_coarse_mesh.geometry.x[:,0],dolfinx_coarse_mesh.geometry.x[:,1]
for i in range(3):
    U_o.vector[i::3] = interp2d(x_f,y_f,U_2.vector[i::3])(x_c,y_c)
p_o.vector = interp2d(x_f,y_f,p.vector)(x_c,y_c)
"""
# Write result as mixed
Space  =dfx.FunctionSpace(dolfinx_fine_mesh,FE_vector_2*FE_scalar)
q = dfx.Function(Space)
U_b,p_b = ufl.split(q)
U_b,p_b=U_2,p
viewer = pet.Viewer().createMPIIO("../cases/nozzle/baseflow/dat_complex/baseflow_S=0.000.dat", 'w', COMM_WORLD)
q.vector.view(viewer)
with XDMFFile(COMM_WORLD, "sanity_check.xdmf", "w") as xdmf:
    xdmf.write_mesh(dolfinx_fine_mesh)
    xdmf.write_function(U_2)