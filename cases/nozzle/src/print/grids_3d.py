import numpy as np


# Custom nonuniform grids
# Extra special grid  for star mode
Xf_es  = np.linspace(0,1,50)
YZf_es = np.linspace(0,1,50)
YZf_es = np.hstack((np.flip(-YZf_es)[:-1],YZf_es)) # Careful of 0 !
XYZf_es = np.meshgrid(Xf_es,YZf_es,YZf_es)
XYZf_es = np.vstack([C.flatten() for C in XYZf_es])

Xr_es  = np.geomspace(5.5,35.2,50)
YZr_es = np.linspace(0,5.6,50)
YZr_es = np.hstack((np.flip(-YZr_es)[:-1],YZr_es)) # Careful of 0 !
XYZr_es = np.meshgrid(Xr_es,YZr_es,YZr_es)
XYZr_es = np.vstack([C.flatten() for C in XYZr_es])

# Grid for modes S,m,St=1,-/+2,.05
Xf_cr  = np.flip(1-np.geomspace(1e-6,1,50))
YZf_cr = np.hstack((np.linspace(0,1,40,endpoint=False), np.linspace(1,1.1,10)))
YZf_cr = np.hstack((np.flip(-YZf_cr)[:-1],YZf_cr)) # Careful of 0 !
XYZf_cr = np.meshgrid(Xf_cr,YZf_cr,YZf_cr)
XYZf_cr = np.vstack([C.flatten() for C in XYZf_cr])

Xr_cr  = np.hstack((np.linspace(.4,1,25,endpoint=False),np.geomspace(1,14.5,25)))
YZr_cr = np.hstack((np.linspace(0,1,25,endpoint=False), np.geomspace(1,2,25)))
YZr_cr = np.hstack((np.flip(-YZr_cr)[:-1],YZr_cr)) # Careful of 0 !
XYZr_cr = np.meshgrid(Xr_cr,YZr_cr,YZr_cr)
XYZr_cr = np.vstack([C.flatten() for C in XYZr_cr])

# Grid for modes S,m,St=0,-/+2,.05
Xr_nr  = np.linspace(1.5,17,50)
YZr_nr = np.linspace(0,1.5,50)
YZr_nr = np.hstack((np.flip(-YZr_nr)[:-1],YZr_nr)) # Careful of 0 !
XYZr_nr = np.meshgrid(Xr_nr,YZr_nr,YZr_nr)
XYZr_nr = np.vstack([C.flatten() for C in XYZr_nr])

# Streaks : long elongated structures, maximum resolution at the nozzle
Xf_sk  = np.hstack((np.flip(1-np.geomspace(.0001,1,35)),np.linspace(1,1.3,10)))
YZf_sk = np.hstack((np.linspace(0,1,25,endpoint=False), np.linspace(1,1.3,25)))
YZf_sk = np.hstack((np.flip(-YZf_sk)[:-1],YZf_sk)) # Careful of 0 !
XYZf_sk = np.meshgrid(Xf_sk,YZf_sk,YZf_sk)
XYZf_sk = np.vstack([C.flatten() for C in XYZf_sk])

Xr_sk  = np.hstack((np.linspace(.5,1.1,20,endpoint=False),np.geomspace(1.1,25,30)))
YZr_sk = np.hstack((np.linspace(0,1,20,endpoint=False), np.geomspace(1,3,30)))
YZr_sk = np.hstack((np.flip(-YZr_sk)[:-1],YZr_sk)) # Careful of 0 !
XYZr_sk = np.meshgrid(Xr_sk,YZr_sk,YZr_sk)
XYZr_sk = np.vstack([C.flatten() for C in XYZr_sk])

# Shorts : short streaks hanging onto the nozzle
Xr_sh  = np.hstack((np.linspace(.5,1.1,20,endpoint=False),np.geomspace(1.1,19,30)))
YZr_sh = np.hstack((np.linspace(0,1,20,endpoint=False), np.geomspace(1,2,30)))
YZr_sh = np.hstack((np.flip(-YZr_sh)[:-1],YZr_sh)) # Careful of 0 !
XYZr_sh = np.meshgrid(Xr_sh,YZr_sh,YZr_sh)
XYZr_sh = np.vstack([C.flatten() for C in XYZr_sh])

# Streaks : long elongated structures away from the nozzle
Xr_st  = np.geomspace(2,20,50)
YZr_st = np.hstack((np.linspace(0,1,20,endpoint=False), np.geomspace(1,3.1,30)))
YZr_st = np.hstack((np.flip(-YZr_st)[:-1],YZr_st)) # Careful of 0 !
XYZr_st = np.meshgrid(Xr_st,YZr_st,YZr_st)
XYZr_st = np.vstack([C.flatten() for C in XYZr_st])

# Swirls : very large and spread out, some distance from the nozzle
Xr_sw  = np.geomspace(5,26.2,50)
YZr_sw = np.linspace(0,3.5,50)
YZr_sw = np.hstack((np.flip(-YZr_sw)[:-1],YZr_sw)) # Careful of 0 !
XYZr_sw = np.meshgrid(Xr_sw,YZr_sw,YZr_sw)
XYZr_sw = np.vstack([C.flatten() for C in XYZr_sw])

# KH : shorter structures nozzle
Xf_kh  = np.linspace(0,1.3,50)
YZf_kh = np.linspace(0,1.3,50)
YZf_kh = np.hstack((np.flip(-YZf_kh)[:-1],YZf_kh)) # Careful of 0 !
XYZf_kh = np.meshgrid(Xf_kh,YZf_kh,YZf_kh)
XYZf_kh = np.vstack([C.flatten() for C in XYZf_kh])

Xr_kh  = np.linspace(.9,12,50)
YZr_kh = np.linspace(0,1.2,50)
YZr_kh = np.hstack((np.flip(-YZr_kh)[:-1],YZr_kh)) # Careful of 0 !
XYZr_kh = np.meshgrid(Xr_kh,YZr_kh,YZr_kh)
XYZr_kh = np.vstack([C.flatten() for C in XYZr_kh])

Xr_kh2  = np.linspace(.9,8,100)
XYZr_kh2 = np.meshgrid(Xr_kh2,YZr_kh,YZr_kh)
XYZr_kh2 = np.vstack([C.flatten() for C in XYZr_kh2])

# Swirls : very large and spread out helix, some distance from the nozzle
n=10
x0,x1=(5,1.05),(16,3)
Xc_sw = np.linspace(6,10,5)
Rc_sw = (Xc_sw-x0[0])*(x1[1]-x0[1])/(x1[0]-x0[0])+x0[1]
Rc_sw = np.hstack((Rc_sw,1.1*Rc_sw))
Xc_sw = np.tile(Xc_sw,2*n)
Yc_sw = np.outer(np.cos(np.linspace(0,2*np.pi,n,endpoint=False)),Rc_sw)
Zc_sw = np.outer(np.sin(np.linspace(0,2*np.pi,n,endpoint=False)),Rc_sw)
XYZc_sw = np.vstack([C.flatten() for C in (Xc_sw,Yc_sw,Zc_sw)])

# Streaks : longer structures, closer to the nozzle
x0,x1=(2.9,.8),(7,1.7)
Xc_st = np.linspace(x0[0],x1[0],5)
Rc_st = (Xc_st-x0[0])*(x1[1]-x0[1])/(x1[0]-x0[0])+x0[1]
Rc_st = np.hstack((Rc_st,1.1*Rc_st))
Xc_st = np.tile(Xc_st,2*n)
Yc_st = np.outer(np.cos(np.linspace(0,2*np.pi,n,endpoint=False)),Rc_st)
Zc_st = np.outer(np.sin(np.linspace(0,2*np.pi,n,endpoint=False)),Rc_st)
XYZc_st = np.vstack([C.flatten() for C in (Xc_st,Yc_st,Zc_st)])