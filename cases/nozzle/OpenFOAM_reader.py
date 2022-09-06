import re, os
import numpy as np
from setup import *
from setup import Re,S
import meshio, ufl, sys #pip3 install --no-binary=h5py h5py meshio
from dolfinx.io import XDMFFile
from petsc4py import PETSc as pet
from scipy.interpolate import griddata
from mpi4py.MPI import COMM_WORLD as comm
from dolfinx.fem import FunctionSpace, Function

sys.path.append('/home/shared/src')

from spy import dirCreator

interpolate=True
cell_type_openfoam="triangle"
cell_type_dolfinx="triangle"
real_mode=False

# Dimensionalised stuff
R,U_M=.1,10
O=np.pi/360 # 0.5°
sin,cos=np.sin(O),np.cos(O)

keys=["Re","S"]
params=[Re,S]

# Read OpenFOAM, write mesh
if comm.rank==0:
    def converter(data,cell_type):
        mesh = meshio.Mesh(points=data.points[:,:2],
                           cells={cell_type: data.get_cells_type(cell_type)})
        meshio.write("nozzle.xdmf", mesh)

    # Searching closest file with respect to setup parameters
    file_names = [f for f in os.listdir(".") if f[-3:]=="xmf"]
    d=np.infty
    for file_name in file_names:
        fd=0
        for param,key in zip(params,keys):
            match = re.search(r'_'+key+r'=(\d*\.?\d*)',file_name)
            param_file = float(match.group(1)) # Take advantage of file format
            fd += abs(param-param_file)
        if fd<d: d,closest_file_name=fd,file_name

    # Read OpenFOAM data
    openfoam_data = meshio.read(closest_file_name)
    openfoam_data.points[:,:2]/=R*cos # Scaling & Plane tilted
    if interpolate:
        # Read coarse data (already plane and non tilted)
        dolfinx_data = meshio.read("nozzle_2D_coarse.msh")
        # Write it out again in a dolfinx friendly format
        converter(dolfinx_data,cell_type_dolfinx)
    else:
        converter(openfoam_data,cell_type_openfoam)
else: openfoam_data=None

openfoam_data = comm.bcast(openfoam_data, root=0) # data available to all but not distributed
# Read it again in dolfinx - now it's a dolfinx object and it's split amongst procs
with XDMFFile(comm, "nozzle.xdmf", "r") as file:
    mesh = file.read_mesh(name="Grid")

# Create FiniteElement, FunctionSpace & Functions
FE_vector  =ufl.VectorElement("CG",mesh.ufl_cell(),1,3)
FE_vector_2=ufl.VectorElement("CG",mesh.ufl_cell(),3,3)
FE_scalar  =ufl.FiniteElement("CG",mesh.ufl_cell(),1)
FE_scalar_2=ufl.FiniteElement("CG",mesh.ufl_cell(),2)
TH=FunctionSpace(mesh,FE_vector_2*FE_scalar_2)
V =FunctionSpace(mesh, FE_vector)
W =FunctionSpace(mesh, FE_scalar)
q = Function(TH)
u = Function(V)
p, nut = Function(W), Function(W)

# Handlers (still useful when !interpolate)
fine_xy=openfoam_data.points[:,:2]
coarse_xy=mesh.geometry.x[:,:2]
def interp(v,reshape=False):
    nv=griddata(fine_xy,v,coarse_xy,'cubic')
    if reshape: return nv.reshape((-1,1))
    return nv

# Dimensionless
uxv,urv,uthv = openfoam_data.point_data['U'].T/U_M
pv   = openfoam_data.point_data['p']/U_M**2
nutv = openfoam_data.point_data['nut']/U_M/R
# Fix orientation
urv,uthv=cos*urv+sin*uthv,-sin*urv+cos*uthv

# Map data onto dolfinx vectors
u.x.array[:]=np.hstack((interp(uxv,1),
                        interp(urv,1),
                        interp(uthv,1))).flatten()
p.x.array[:]  =interp(pv)
nut.x.array[:]=interp(nutv)

# Write result as mixed
us,ps=q.split()
us.interpolate(u)
ps.interpolate(p)
"""
# Save pretty graphs
for f in ['us','ps','nut']:
    with XDMFFile(comm, "sanity_check_"+f+".xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        eval("xdmf.write_function("+f+")")
"""
# Write turbulent viscosity separately
dirCreator("./baseflow")
dirCreator("./baseflow/nut")
if real_mode: dirCreator("./baseflow/dat_real")
else:         dirCreator("./baseflow/dat_complex")
viewer = pet.Viewer().createMPIIO(f"./baseflow/nut/nut_S={S:.3f}_Re={Re:d}_n={comm.size:d}.dat", 'w', comm)
nut.vector.view(viewer)
if real_mode:
    viewer = pet.Viewer().createMPIIO(f"./baseflow/dat_real/baseflow_S={S:.3f}_Re={Re:d}_n={comm.size:d}.dat", 'w', comm)
else:
    viewer = pet.Viewer().createMPIIO(f"./baseflow/dat_complex/baseflow_S={S:.3f}_Re={Re:d}_n={comm.size:d}.dat", 'w', comm)
q.vector.view(viewer)