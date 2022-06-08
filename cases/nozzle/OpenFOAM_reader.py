import meshio, ufl #pip3 install --no-binary=h5py h5py meshio
import numpy as np
import dolfinx as dfx
from dolfinx.io import XDMFFile
from mpi4py.MPI import COMM_WORLD as comm
from petsc4py import PETSc as pet
#from scipy.interpolate import interp2d

p0=comm.rank==0

# Dimensionalised stuff
R,U_M=.1,10
o=np.pi/360
s,c=np.sin(o),np.cos(o)
if p0:
    # Read mesh and point data
    openfoam_data = meshio.read("front.xmf")
    # Write it out again
    ps=openfoam_data.points[:,:2]/R # Scaling
    ps[:,1]/=c # Plane tilted
    cs = openfoam_data.get_cells_type("quad")
    dolfinx_fine_mesh = meshio.Mesh(points=ps, cells={"quad": cs})
    meshio.write("nozzle.xdmf", dolfinx_fine_mesh)
else: openfoam_data=None
openfoam_data = comm.bcast(openfoam_data, root=0) # openfoam_data available to all but not distributed
# Read it again in dolfinx - now it's a dolfinx object
with XDMFFile(comm, "nozzle.xdmf", "r") as file:
    dolfinx_fine_mesh = file.read_mesh(name="Grid")

# Create FiniteElement, FunctionSpace & Functions
FE_vector_1=ufl.VectorElement("Lagrange",dolfinx_fine_mesh.ufl_cell(),1,3)
FE_vector_2=ufl.VectorElement("Lagrange",dolfinx_fine_mesh.ufl_cell(),2,3)
FE_scalar=ufl.FiniteElement("Lagrange",dolfinx_fine_mesh.ufl_cell(),1)
V1=dfx.FunctionSpace(dolfinx_fine_mesh,FE_vector_1)
V2=dfx.FunctionSpace(dolfinx_fine_mesh,FE_vector_2)
W=dfx.FunctionSpace(dolfinx_fine_mesh,FE_scalar)
u1, u2 = dfx.Function(V1), dfx.Function(V2)
p, nut = dfx.Function(W), dfx.Function(W)

# Global indexes owned locally
ids = dolfinx_fine_mesh.geometry.input_global_indices
# Map OpenFOAM data directy onto dolfinx vectors
u1.x.array[:]  = openfoam_data.point_data['U'][  ids,:].flatten()/U_M
p.x.array[:]   = openfoam_data.point_data['p'][  ids]/U_M**2
nut.x.array[:] = openfoam_data.point_data['nut'][ids]/U_M/R
# Fix orientation
Uy,Uz=u1.vector[1::3],u1.vector[2::3]
Uy,Uz=c*Uy+s*Uz,-s*Uy+c*Uz
u1.vector[1::3],u1.vector[2::3]=Uy,Uz
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
u2.interpolate(u1)
with XDMFFile(comm, "sanity_check_parallel.xdmf", "w") as xdmf:
    xdmf.write_mesh(dolfinx_fine_mesh)
    xdmf.write_function(u2)
# Write result as mixed
TH = dfx.FunctionSpace(dolfinx_fine_mesh,FE_vector_2*FE_scalar)
_, dofs_U = TH.sub(0).collapse(collapsed_dofs=True)
_, dofs_p = TH.sub(1).collapse(collapsed_dofs=True)
q = dfx.Function(TH)
q.x.array[dofs_U]=u2.x.array
q.x.array[dofs_p]=p.x.array

# Write turbulent viscosity separately
viewer = pet.Viewer().createMPIIO("./baseflow/nut/nut_S=0.000.dat", 'w', comm)
nut.vector.view(viewer)
viewer = pet.Viewer().createMPIIO("./baseflow/dat_complex/baseflow_S=0.000.dat", 'w', comm)
q.vector.view(viewer)