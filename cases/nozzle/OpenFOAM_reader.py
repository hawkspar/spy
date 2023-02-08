import numpy as np
import meshio, ufl, sys #pip3 install h5py meshio
from setup import *
from dolfinx.io import XDMFFile
from scipy.interpolate import griddata
from mpi4py.MPI import COMM_WORLD as comm
from dolfinx.fem import FunctionSpace, Function

sys.path.append('/home/shared/src')

from spy import SPY, dirCreator, meshConvert, findStuff, saveStuff

p0=comm.rank==0
sanity_check=True#False
convert=False

# Relevant parameters
Res=[1000,10000,100000,400000]
Ss=[0,.2,.4,.6,.8,1]

# Convert mesh
if convert: meshConvert("baseflow")
spy=SPY(params,datapath,'baseflow',direction_map)

# Dimensionalised stuff
L,H=50.5,10
O=np.pi/360 # 0.5°
sin,cos=np.sin(O),np.cos(O)

for Re in Res:
    for S in Ss:
        if S!=0 and Re!=400000: continue
        # Read OpenFOAM, write mesh
        if p0:
            # Searching closest file with respect to setup parameters
            closest_file_name=findStuff("./baseflow/OpenFOAM/",{'S':S,'Re':Re}, lambda f: f[-3:]=="xmf",False)
            # Read OpenFOAM data
            openfoam_data = meshio.read(closest_file_name)
            print("Loaded "+closest_file_name+" successfully !", flush=True)

            # Read cell_centers
            center_points = openfoam_data.cell_data['CellCenters'][0]
            fine_xy = np.vstack((openfoam_data.points,center_points)) # Regroup all data coordinates
            fine_xy[:,1]/=cos # Plane tilted

            # Reducing problem size (coarse mesh is also smaller)
            msk = np.all(fine_xy[:,:2]<1.1*np.array([L,H]),1)

            fine_xy=fine_xy[msk,:2]

            # Dimensionless
            uxv,urv,uthv = np.vstack((openfoam_data.point_data['U'],openfoam_data.cell_data['U'][0]))[msk,:].T
            pv   = np.hstack((openfoam_data.point_data['p'],  openfoam_data.cell_data['p'][0])  )[msk]
            nutv = np.hstack((openfoam_data.point_data['nut'],openfoam_data.cell_data['nut'][0]))[msk]

        else: uxv,urv,uthv,pv,nutv,fine_xy=None,None,None,None,None,None

        # Data available to all but not distributed
        uxv  = comm.bcast(uxv,  root=0)
        urv  = comm.bcast(urv,  root=0)
        uthv = comm.bcast(uthv, root=0)
        pv   = comm.bcast(pv,   root=0)
        nutv = comm.bcast(nutv, root=0)
        fine_xy = comm.bcast(fine_xy, root=0)

        # Handlers (still useful when !interpolate)
        def interp(v,x): return griddata(fine_xy,v,x[:2,:].T,'cubic')
        # Fix orientation
        urv,uthv=cos*urv+sin*uthv,-sin*urv+cos*uthv
        # Fix no swirl edge case
        if S==0: uthv[:]=0

        # Handlers
        U,P=spy.Q.split()
        Nu=spy.Nu

        # Map data onto dolfinx vectors
        U.sub(0).interpolate(lambda x: interp(uxv, x))
        U.sub(1).interpolate(lambda x: interp(urv, x)*(x[1]>params['atol'])) # Enforce u_r=u_th=0 at r=0
        U.sub(2).interpolate(lambda x: interp(uthv,x)*(x[1]>params['atol']))
        P.interpolate( lambda x: interp(pv,  x))
        Nu.interpolate(lambda x: interp(nutv,x))
        # Fix negative eddy viscosity
        Nu.x.array[Nu.x.array<0] = 0

        # Save
        #spy.saveBaseflow(Re,S)
        dirCreator(spy.baseflow_path)
        saveStuff(spy.nut_path,f'nut_S={S:.1f}_Re={Re:d}'.replace('.',','),Nu)
        spy.printStuff(spy.baseflow_path+'print_OpenFOAM/',f"u_Re={Re:d}_S={S:.1f}".replace('.',','),U)