import meshio #pip3 install h5py meshio
import numpy as np
from setup import *
from scipy.interpolate import griddata
from mpi4py.MPI import COMM_WORLD as comm
from helpers import dirCreator, meshConvert, findStuff

sanity_check=True#False
convert=False

# Relevant parameters
Res=[1000,10000,100000,200000]
Ss=np.linspace(0,1.6,17)
# Convert mesh
if convert: meshConvert(base_mesh)

# Dimensionalised stuff
r=1.2
O=np.pi/360 # 0.5°
sin,cos=np.sin(O),np.cos(O)
H,L=15,50 # Actual dolfinx mesh size

# Handlers
U,P=spyb.Q.split()

for Re in Res:
    for S in Ss:
        if S!=0 and Re!=200000: continue
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
            msk = (fine_xy[:,0]<r*L)*(fine_xy[:,1]<r*H)
            fine_xy=fine_xy[msk,:2]

            # Dimensionless
            uxv,urv,uthv = np.vstack((openfoam_data.point_data['U'],openfoam_data.cell_data['U'][0]))[msk,:].T
            pv = np.hstack((openfoam_data.point_data['p'],openfoam_data.cell_data['p'][0]))[msk]
            nutv = np.hstack((openfoam_data.point_data['nut'],openfoam_data.cell_data['nut'][0]))[msk]
        else: uxv,urv,uthv,pv,nutv,fine_xy=None,None,None,None,None,None

        # Data available to all but not distributed
        uxv  = comm.bcast(uxv,  root=0)
        urv  = comm.bcast(urv,  root=0)
        uthv = comm.bcast(uthv, root=0)
        pv = comm.bcast(pv, root=0)
        nutv = comm.bcast(nutv, root=0)
        fine_xy = comm.bcast(fine_xy, root=0)

        # Handlers (still useful when !interpolate, uses splines)
        def interp(v,x): return griddata(fine_xy,v,x[:2,:].T,'linear')
        # Fix orientation
        urv,uthv=cos*urv+sin*uthv,-sin*urv+cos*uthv
        # Fix no swirl edge case
        if S==0: uthv[:]=0

        # Map data onto dolfinx vectors
        U.sub(0).interpolate(lambda x: interp(uxv, x))
        U.sub(1).interpolate(lambda x: interp(urv, x)*(x[1]>params['atol'])) # Enforce u_r=u_th=0 at r=0
        U.sub(2).interpolate(lambda x: interp(uthv,x)*(x[1]>params['atol']))
        P.interpolate(lambda x: interp(pv,x))
        spyb.Nu.interpolate(lambda x: interp(nutv,x))
        # Laplace smoothing
        spyb.smoothenU(1e-6)
        spyb.smoothenP(1e-6)
        spyb.smoothenNu(1e-4)

        # Save
        dirCreator(spyb.baseflow_path)
        save_string=f"_Re={Re:d}_S={S:.1f}".replace('.',',')
        spyb.saveBaseflow(Re,S,True)
        spyb.printStuff(spyb.baseflow_path+'print_OpenFOAM/',"nu"+save_string,spyb.Nu)
        spyb.printStuff(spyb.baseflow_path+'print_OpenFOAM/',"u" +save_string,U)