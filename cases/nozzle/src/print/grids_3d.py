import numpy as np

n=50

# Handler
def XYZ(X,YZ):
	if not type(YZ)==np.ndarray: YZ=np.linspace(0,YZ,n)
	YZ = np.hstack((np.flip(-YZ)[:-1],YZ)) # Careful of 0 !
	XYZ = np.meshgrid(X,YZ,YZ)
	return np.vstack([C.flatten() for C in XYZ])

# Very simple mesh for structs close to nozzle top at S,m,St=0,0,0
X  = np.geomspace(1e-6,1,10)
YZ = np.linspace(0,1,5)
XYZ_pt = XYZ(X,YZ)

# Custom nonuniform grids
# Very simple mesh for structs close to nozzle top at S,m,St=0,0,0
Xf_cl  = np.linspace(0,1,n)
XYZf_cl = XYZ(Xf_cl,1.1)

Xr_cl  = np.linspace(0,1.1,n)
XYZr_cl = XYZ(Xr_cl,1.5)

# Extra special grid for star mode
Xf_es  = np.linspace(0,1.01,70)
YZf_es  = np.linspace(0,1.3,70)
XYZf_es = XYZ(Xf_es,YZf_es)

Xr_es  = np.geomspace(7.5,47,70)
YZr_es = np.linspace(0,8,70)
XYZr_es = XYZ(Xr_es,YZr_es)

# Grid for modes S,m,St=1,-/+2,0
Xf_cr  = np.flip(1-np.geomspace(1e-6,1,n))
YZf_cr = np.hstack((np.linspace(0,1,40,endpoint=False), np.linspace(1,1.1,10)))
XYZf_cr = XYZ(Xf_cr,YZf_cr)

Xr_cr  = np.hstack((np.linspace(.4,1,n//2,endpoint=False),np.geomspace(1,14.5,n//2)))
YZr_cr = np.hstack((np.linspace(0, 1,n//2,endpoint=False),np.geomspace(1, 2,  n//2)))
XYZr_cr = XYZ(Xr_cr,YZr_cr)

# Grid for modes S,m,St=0,-/+2,0
Xf_nr  = np.linspace(0,1,n)
XYZf_nr = XYZ(Xf_nr,1.15)

Xr_nr  = np.geomspace(1e-6,50,n)
XYZr_nr = XYZ(Xr_nr,4)

# Streaks : long elongated structures away from the nozzle
Xr_st  = np.geomspace(2,20,n)
YZr_st = np.hstack((np.linspace(0,1,20,endpoint=False), np.geomspace(1,3.1,30)))
XYZr_st = XYZ(Xr_st,YZr_st)

# Swirls : very large and spread out, some distance from the nozzle
Xr_sw  = np.geomspace(5,26.2,n)
XYZr_sw = XYZ(Xr_sw,3.5)

# KH : shorter structures nozzle
Xf_kh  = np.linspace(0,1.1,60)
YZf_kh = np.linspace(0,1.01,70)
XYZf_kh = XYZ(Xf_kh,YZf_kh)

Xr_kh  = np.linspace(1.5,12,70)
YZr_kh = np.linspace(0,1.5,70)
XYZr_kh = XYZ(Xr_kh,YZr_kh)

# Cylindrical meshes for baseflow quivers
n=10
def XYZc(x0,x1):
	X = np.linspace(x0[0],x1[0],n//2)
	R = (X-x0[0])/(x1[0]-x0[0])*(x1[1]-x0[1])+x0[1]
	R = np.hstack((R,1.1*R))
	X = np.tile(X,2*n)
	Y = np.outer(np.cos(np.linspace(0,2*np.pi,n,endpoint=False)),R)
	Z = np.outer(np.sin(np.linspace(0,2*np.pi,n,endpoint=False)),R)
	return np.vstack([C.flatten() for C in (X,Y,Z)])

# Swirls : very large and spread out helix, some distance from the nozzle
x0,x1=(5,1.05),(16,3)
XYZc_sw = XYZc(x0,x1)

# Streaks : longer structures, closer to the nozzle
x0,x1=(2.9,.8),(7,1.7)
XYZc_st = XYZc(x0,x1)

# Streaks : longer structures, closer to the nozzle
x0,x1=(5,1.1),(26,4.7)
XYZc_es = XYZc(x0,x1)