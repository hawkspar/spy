import re, os
import numpy as np
import meshio, ufl, sys #pip3 install --no-binary=h5py h5py meshio
from setup import Re, S
from dolfinx.io import XDMFFile
from scipy.interpolate import griddata
from mpi4py.MPI import COMM_WORLD as comm
from dolfinx.fem import FunctionSpace, Function, Expression

sys.path.append('/home/shared/src')

from spy import dirCreator, meshConvert

p0=comm.rank==0

interpolate=True
sanity_check=True
cell_type="triangle"

# Dimensionalised stuff
R,U_M=.1,10
L,H=36,5
O=np.pi/360 # 0.5°
sin,cos=np.sin(O),np.cos(O)

# Read OpenFOAM, write mesh
if p0:
    # Searching closest file with respect to setup parameters
    file_names = [f for f in os.listdir(".") if f[:5]=="front" and f[-3:]=="xmf"]
    d = np.infty
    for file_name in file_names:
        fd = 0
        for param,key in zip([Re,S],["Re","S"]):
            match = re.search(r'_'+key+r'=(\d*\.?\d*)',file_name)
            param_file = float(match.group(1)) # Take advantage of file format
            fd += abs(param-param_file)
        if fd<d: d,closest_file_name=fd,file_name
    
    # Read OpenFOAM data
    openfoam_data = meshio.read(closest_file_name)
    print("Loaded "+closest_file_name+" successfully !", flush=True)

    # Read cell_centers
    center_points = openfoam_data.cell_data['CellCenters'][0]

    fine_xy = np.vstack((openfoam_data.points,center_points))
    fine_xy[:,:2]/=R # Scaling & Plane tilted
    fine_xy[:, 1]/=cos

    # Reducing problem size (coarse mesh is also smaller)
    msk = np.logical_and(fine_xy[:,0]<L,fine_xy[:,1]<H)
    fine_xy=fine_xy[msk,:2]

    # Dimensionless
    uxv,urv,uthv = np.vstack((openfoam_data.point_data['U'],openfoam_data.cell_data['U'][0]))[msk,:].T/U_M
    pv   = np.hstack((openfoam_data.point_data['p'],  openfoam_data.cell_data['p'][0])  )[msk]/U_M**2
    nutv = np.hstack((openfoam_data.point_data['nut'],openfoam_data.cell_data['nut'][0]))[msk]/U_M/R

    # Convert mesh
    if interpolate: meshConvert("nozzle_2D_coarse","nozzle",cell_type)
    # Important to ensure consistancy in partioning
    else:           meshConvert("nozzle_2D","nozzle",cell_type)
else: uxv,urv,uthv,pv,nutv,fine_xy=None,None,None,None,None,None

uxv = comm.bcast(uxv, root=0) # data available to all but not distributed
urv = comm.bcast(urv, root=0)
uthv = comm.bcast(uthv, root=0)
pv = comm.bcast(pv, root=0)
nutv = comm.bcast(nutv, root=0)
fine_xy = comm.bcast(fine_xy, root=0)
# Read it again in dolfinx - now it's a dolfinx object and it's split amongst procs
with XDMFFile(comm, "nozzle.xdmf", "r") as file:
    mesh = file.read_mesh(name="Grid")

# Create FiniteElement, FunctionSpace & Functions
FE_vector=ufl.VectorElement("CG",mesh.ufl_cell(),2,3)
FE_scalar=ufl.FiniteElement("CG",mesh.ufl_cell(),1)
V =FunctionSpace(mesh, FE_vector)
W =FunctionSpace(mesh, FE_scalar)
U, P, nut = Function(V), Function(W), Function(W)

# Handlers (still useful when !interpolate)
def interp(v,x): return griddata(fine_xy,v,x[:2,:].T,'cubic')
# Fix orientation
urv,uthv=cos*urv+sin*uthv,-sin*urv+cos*uthv
# Fix no swirl edge case
if S==0: uthv[:]=0
# Fix negative eddy visocisty
nutv[nutv<0] = 0

# Map data onto dolfinx vectors
U.sub(0).interpolate(lambda x: interp(uxv, x))
U.sub(1).interpolate(lambda x: interp(urv, x))
U.sub(2).interpolate(lambda x: interp(uthv,x))
U.x.scatter_forward()
P.interpolate(  lambda x: interp(pv,   x))
P.x.scatter_forward()
nut.interpolate(lambda x: interp(nutv, x))
nut.x.scatter_forward()

# Save pretty graphs
if sanity_check:
    for f in ['U','P','nut']:
        with XDMFFile(comm, "sanity_check_"+f+"_reader.xdmf", "w") as xdmf:
            xdmf.write_mesh(mesh)
            eval("xdmf.write_function("+f+")")

# Write all functions separately
pre="./baseflow"
dirCreator(pre)
dirCreator(pre+"/u/")
dirCreator(pre+"/p/")
dirCreator(pre+"/nut/")

app=f"_S={S:.3f}_Re={Re:d}_n={comm.size:d}_p={comm.rank:d}"

# Save
np.save(pre+"/u/u"+app,    U.x.array)
np.save(pre+"/p/p"+app,    P.x.array)
np.save(pre+"/nut/nut"+app,nut.x.array)