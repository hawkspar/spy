# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from print_3d import *

Ss_ref = [1]
ms_ref = [-2,2]#,0,2]
Sts_ref = [.025]#,.5]

# Swirls : very large and spread out helix, some distance from the nozzle
n=10
Xc_sw = np.linspace(6,10,5)
x0,x1=(5,1.05),(16,4.3)
Rc_sw = (Xc_sw-x0[0])/(x1[0]-x0[0])*x1[1]+x0[1]
Xc_sw = np.tile(Xc_sw,n)
Yc_sw = np.outer(np.cos(np.linspace(0,2*np.pi,n,endpoint=False)),Rc_sw)
Zc_sw = np.outer(np.sin(np.linspace(0,2*np.pi,n,endpoint=False)),Rc_sw)
XYZc_sw = np.vstack([C.flatten() for C in (Xc_sw,Yc_sw,Zc_sw)])

# Streaks : longer structures, closer to the nozzle
Xc_st = np.linspace(3,10,5)
x0,x1=(3,1),(10,2.5)
Rc_st = (Xc_st-x0[0])/(x1[0]-x0[0])*x1[1]+x0[1]
Xc_st = np.tile(Xc_st,n)
Yc_st = np.outer(np.cos(np.linspace(0,2*np.pi,n,endpoint=False)),Rc_st)
Zc_st = np.outer(np.sin(np.linspace(0,2*np.pi,n,endpoint=False)),Rc_st)
XYZc_st = np.vstack([C.flatten() for C in (Xc_st,Yc_st,Zc_st)])

for S in Ss_ref:
	spyb.loadBaseflow(Re,S,False)
	for m in ms_ref:
		for St in Sts_ref:
			file_name=dir+f"Re={2*Re:d}_S={S:.1f}_m={m:d}_St={2*St}".replace('.',',') # Usual Re & St based on D
			if p0: print(f"Currently beautifying (Re,S,m,St)=({Re},{S:.1f},{m},{St})",flush=True)
			if isfile(file_name):
				if p0: print("Found an html file, moving on...",flush=True)
				continue
			if St > .5 or m == 0:
				#isos_f=spyp.computeIsosurfaces(m,XYZf_kh,.1,spyp.readMode("forcing", Re,S,m,St),2,'Earth', "axial forcing")
				isos_r=spyp.computeIsosurfaces(m,XYZr_kh,.1,spyp.readMode("response",Re,S,m,St),2,'Picnic',"axial response")
			elif St == Sts_ref[0] and S == 0:
				#isos_f=spyp.computeIsosurfaces(m,XYZf_sk,.1,spyp.readMode("forcing", Re,S,m,St),2,'Earth', "axial forcing")
				isos_r=spyp.computeIsosurfaces(m,XYZr_sh,.1,spyp.readMode("response",Re,S,m,St),2,'Picnic',"axial response")
			elif St == Sts_ref[0] and S < .5:
				#isos_f=spyp.computeIsosurfaces(m,XYZf_sk,.1,spyp.readMode("forcing", Re,S,m,St),2,'Earth', "axial forcing")
				isos_r=spyp.computeIsosurfaces(m,XYZr_sk,.1,spyp.readMode("response",Re,S,m,St),2,'Picnic',"axial response")
			elif St == Sts_ref[0] and m > 0:
				quiv_U=spyb.computeQuiver(XYZc_st,"Greens")
				isos_r,isos_f=[],[]
				for coord in range(3):
					isos_f.append(spyp.computeIsosurfaces(m,XYZf_cr,.1,spyp.readMode("forcing", Re,S,m,St,coord),2,'Earth', "axial forcing"))
					isos_r.append(spyp.computeIsosurfaces(m,XYZr_st,.1,spyp.readMode("response",Re,S,m,St,coord),2,'Picnic',"axial response"))
			elif St == Sts_ref[0] and m < 0:
				quiv_U=spyb.computeQuiver(XYZc_sw,"Greens")
				isos_r,isos_f=[],[]
				for coord in range(3):
					isos_f.append(spyp.computeIsosurfaces(m,XYZf_cr,.1,spyp.readMode("forcing", Re,S,m,St,coord),2,'Earth', "axial forcing"))
					isos_r.append(spyp.computeIsosurfaces(m,XYZr_sw,.1,spyp.readMode("response",Re,S,m,St,coord),2,'Picnic',"axial response"))
			elif St == Sts_ref[-1]:
				#isos_f=spyp.computeIsosurfaces(m,XYZf_kh,.1,spyp.readMode("forcing", Re,S,m,St),2,'Earth', "axial forcing")
				isos_r=spyp.computeIsosurfaces(m,XYZr_kh,.1,spyp.readMode("response",Re,S,m,St),2,'Picnic',"axial response")
			else:
				#isos_f=spyp.computeIsosurfaces(m,XYZf_cr,.1,spyp.readMode("forcing", Re,S,m,St),2,'Earth', "axial forcing")
				isos_r=spyp.computeIsosurfaces(m,XYZr_nr,.1,spyp.readMode("response",Re,S,m,St),2,'Picnic',"axial response")
			# Animation
			if p0:
				for coord in range(3):
					fig = go.Figure(data=[isos_f[coord][0], isos_r[coord][0], quiv_U, up_nozzle, down_nozzle])
					fig.update_coloraxes(showscale=False)
					fig.update_layout(scene_aspectmode='data')
					fig.write_html(file_name+f"_dir={list(direction_map.keys())[coord]}.html")