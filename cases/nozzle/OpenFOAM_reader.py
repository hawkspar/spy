import meshio, ufl #pip3 install --no-binary=h5py h5py meshio
import numpy as np
import dolfinx as dfx
from dolfinx.io import XDMFFile
from mpi4py.MPI import COMM_WORLD
from petsc4py import PETSc as pet
#from scipy.interpolate import interp2d

# Dimensionalised stuff
R,U_M=.1,10
o=np.pi/360
s,c=np.sin(o),np.cos(o)
# Read mesh and point data
openfoam_mesh = meshio.read("front.xmf")
# Write it out again
pts=openfoam_mesh.points[:,:2]/R
pts[:,1]/=c
ces = openfoam_mesh.get_cells_type("quad")
dolfinx_fine_mesh = meshio.Mesh(points=pts, cells={"quad": ces}) # Note the adimensioning
meshio.write("nozzle.xdmf", dolfinx_fine_mesh)
# Read it again in dolfinx
with XDMFFile(COMM_WORLD, "nozzle.xdmf", "r") as file:
    dolfinx_fine_mesh = file.read_mesh(name="Grid")
# Create FiniteElement, FunctionSpace & Functions
FE_vector_1=ufl.VectorElement("Lagrange",dolfinx_fine_mesh.ufl_cell(),1,3)
FE_vector_2=ufl.VectorElement("Lagrange",dolfinx_fine_mesh.ufl_cell(),2,3)
FE_scalar=ufl.FiniteElement("Lagrange",dolfinx_fine_mesh.ufl_cell(),1)
V_1=dfx.FunctionSpace(dolfinx_fine_mesh,FE_vector_1)
V_2=dfx.FunctionSpace(dolfinx_fine_mesh,FE_vector_2)
W=dfx.FunctionSpace(dolfinx_fine_mesh,FE_scalar)
U_1 = dfx.Function(V_1)
U_2 = dfx.Function(V_2)
p = dfx.Function(W)
nut = dfx.Function(W)
# Jiggle indexing
idcs = np.argsort(dolfinx_fine_mesh.geometry.input_global_indices).astype('int32')
vec_idcs = np.repeat(3*idcs,3)
vec_idcs[1::3]+=1
vec_idcs[2::3]+=2
# Map OpenFOAM data directy onto dolfinx vectors
U_1.vector[vec_idcs] = openfoam_mesh.point_data['U'].flatten()/U_M
p.vector[idcs] = openfoam_mesh.point_data['p']/U_M**2
nut.vector[idcs] = openfoam_mesh.point_data['nut']/U_M/R
# Fix orientation
Uy,Uz=U_1.vector[1::3],U_1.vector[2::3]
Uy,Uz=c*Uy+s*Uz,-s*Uy+c*Uz
U_1.vector[1::3],U_1.vector[2::3]=Uy,Uz
"""
# Read mesh and point data
coarse_mesh = meshio.read("../cases/nozzle/nozzle_coarse.msh")
# Write it out again
cells = coarse_mesh.get_cells_type("quad") 
dolfinx_coarse_mesh = meshio.Mesh(points=coarse_mesh.points[:,:2], cells={"quad": cells})
meshio.write("../cases/nozzle/nozzle_coarse.xdmf", dolfinx_coarse_mesh)
# Read smaller mesh in dolfinx
with XDMFFile(COMM_WORLD, "../cases/nozzle/nozzle_coarse.xdmf", "r") as file:
    dolfinx_coarse_mesh = file.read_mesh(name="Grid")
# Create coarser Functions
FE_vector_1=ufl.VectorElement("Lagrange",dolfinx_coarse_mesh.ufl_cell(),1,3)
FE_vector_2=ufl.VectorElement("Lagrange",dolfinx_coarse_mesh.ufl_cell(),2,3)
FE_scalar=ufl.FiniteElement("Lagrange",dolfinx_coarse_mesh.ufl_cell(),1)
V_1=dfx.FunctionSpace(dolfinx_coarse_mesh,FE_vector_1)
V_2=dfx.FunctionSpace(dolfinx_coarse_mesh,FE_vector_2)
W=dfx.FunctionSpace(dolfinx_coarse_mesh,FE_scalar)
U_1 = dfx.Function(V_1)
U_2 = dfx.Function(V_2)
p_o = dfx.Function(W)
nut_o = dfx.Function(W)
# Interpolate results on coarser mesh
x_f,y_f=dolfinx_fine_mesh.geometry.x[:,0],  dolfinx_fine_mesh.geometry.x[:,1]
x_c,y_c=dolfinx_coarse_mesh.geometry.x[:,0],dolfinx_coarse_mesh.geometry.x[:,1]
for i in range(3):
    U_1.vector[i::3] = interp2d(x_f,y_f,U.vector[i::3].real)
p_o.vector = interp2d(x_f,y_f,p.vector.real)(x_c,y_c)
nut_o.vector = interp2d(x_f,y_f,nut.vector.real)(x_c,y_c)
"""
# Interpolation to higher order
U_2.interpolate(U_1)
# Write turbulent viscosity separately
viewer = pet.Viewer().createMPIIO("./baseflow/nut/nut_S=0.000.dat", 'w', COMM_WORLD)
nut.vector.view(viewer)
# Write result as mixed
Space = dfx.FunctionSpace(dolfinx_fine_mesh,FE_vector_2*FE_scalar)
q = dfx.Function(Space)
_, map_U = Space.sub(0).collapse(collapsed_dofs=True)
_, map_p = Space.sub(1).collapse(collapsed_dofs=True)
q.vector[map_U]=U_2.vector
q.vector[map_p]=p.vector
viewer = pet.Viewer().createMPIIO("./baseflow/dat_complex/baseflow_S=0.000.dat", 'w', COMM_WORLD)
q.vector.view(viewer)