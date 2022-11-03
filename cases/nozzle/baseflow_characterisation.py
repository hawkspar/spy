import meshio #pip3 install --no-binary=h5py h5py meshio
import numpy as np
from itertools import product
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

# Dimensionalised stuff
C=np.cos(np.pi/360) # 0.5°
u_co=.05
Q=(1-u_co)/(1+u_co)

# Read OpenFOAM
# Read mesh and point data
openfoam_data = meshio.read("front_Re=400000_S=0.xmf")
xy=openfoam_data.points[:,:2]/C # Scaling & Plane tilted

# Dimensionless
ud,_,_ = openfoam_data.point_data['U'].T

n = 100
X = np.linspace(0,60,n)
target_xy = np.zeros((n,2))
target_xy[:,0] = X

def interp(v,target_xy): return griddata(xy,v,target_xy,'cubic') 

u = interp(ud,target_xy)

plt.plot(X, u)
plt.xlabel(r'$x$')
plt.ylabel(r'$U_{x,r=0}$')
plt.savefig("Ur0(x).png")
plt.close()

R = np.linspace(0,15,n)
target_xy = np.array(list(product(X,R)))

u = interp(ud,target_xy).reshape((n,n))
R_n = R.reshape([1,-1])
ths = np.trapz((u-u_co)*(1-u)*R_n,R)

plt.plot(X, ths)
plt.xlabel(r'$x$')
plt.ylabel(r'$\theta$')
plt.savefig("theta(x).png")
plt.close()

sgths = np.gradient(ths,X)*u_co/Q

plt.plot(X, sgths)
plt.xlabel(r'$x$')
plt.ylabel(r'$u_{co}/Q d_x\theta$')
plt.savefig("Cdxtheta(x).png")