# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from modes_3d import *

Ss_ref = [1]
ms_ref = [-2,2]#,0,2]
Sts_ref = [.025]#,.5]

directions=list(direction_map.keys())

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

for S in Ss_ref:
	spyb.loadBaseflow(Re,S,False)
	for m in ms_ref:
		for St in Sts_ref:
			file_name=dir+f"Re={Re:d}_S={S:.1f}_m={m:d}_St={St:.4e}".replace('.',',') # Usual Re & St based on D
			dat={"Re":Re,"S":S,"m":m,"St":St}
			if p0: print(f"Currently beautifying (Re,S,m,St)=({Re},{S:.1f},{m},{St:.2f})",flush=True)
			if isfile(file_name+"_dir=x.html") and isfile(file_name+"_dir=r.html") and isfile(file_name+"_dir=th.html"):
				if p0: print("Found html files, moving on...",flush=True)
				continue
			if St > .5 or m == 0:
				#isos_f=spyp.computeIsosurfaces("forcing",dat,XYZf_kh,.1,2,'Earth', "axial forcing")
				isos_r=spyp.computeIsosurfaces("response",dat,XYZr_kh,.1,2,'Picnic',"axial response")
			elif St == Sts_ref[0] and S == 0:
				#isos_f=spyp.computeIsosurfaces("forcing",dat,XYZf_sk,.1,2,'Earth', "axial forcing")
				isos_r=spyp.computeIsosurfaces("response",dat,XYZr_sh,.1,2,'Picnic',"axial response")
			elif St == Sts_ref[0] and S < .5:
				#isos_f=spyp.computeIsosurfaces("forcing",dat,XYZf_sk,.1,2,'Earth', "axial forcing")
				isos_r=spyp.computeIsosurfaces("response",dat,XYZr_sk,.1,2,'Picnic',"axial response")
			elif St == Sts_ref[0] and m > 0:
				quiv_U=spyb.computeQuiver(XYZc_st,"Greens")
				isos_f=spyp.computeIsosurfaces("forcing", dat,XYZf_cr,.1,1,'Earth', "forcing",True)
				isos_r=spyp.computeIsosurfaces("response",dat,XYZr_st,.1,1,'Picnic',"response",True)
			elif St == Sts_ref[0] and m < 0:
				quiv_U=spyb.computeQuiver(XYZc_sw,"Greens")
				isos_f=spyp.computeIsosurfaces("forcing", dat,XYZf_cr,.1,1,'Earth', "forcing",True)
				isos_r=spyp.computeIsosurfaces("response",dat,XYZr_sw,.1,1,'Picnic',"response",True)
			elif St == Sts_ref[-1]:
				#isos_f=spyp.computeIsosurfaces("forcing",dat,XYZf_kh,.1,2,'Earth', "axial forcing")
				isos_r=spyp.computeIsosurfaces("response",dat,XYZr_kh,.1,2,'Picnic',"axial response")
			else:
				#isos_f=spyp.computeIsosurfaces("forcing",dat,XYZf_cr,.1,2,'Earth', "axial forcing")
				isos_r=spyp.computeIsosurfaces("response",dat,XYZr_nr,.1,2,'Picnic',"axial response")
			# Animation
			if p0:
				for coord in range(3):
					fig = go.Figure(data=[isos_f[coord][0], isos_r[coord][0], quiv_U, up_nozzle, down_nozzle])
					fig.update_coloraxes(showscale=False)
					fig.update_layout(scene_aspectmode='data')
					fig.write_html(f"{file_name}_dir={directions[coord]}_cone.html")
					if p0: print(f"Wrote {file_name}_dir={directions[coord]}_cone.html",flush=True)