# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from sys import path
from os.path import isfile
import plotly.graph_objects as go

path.append('/home/shared/cases/nozzle/src/')

from setup import *
from spyp import SPYP # Must be after setup
from helpers import dirCreator

spyp = SPYP(params,data_path,pert_mesh,direction_map)

# Actual plotting
dir=spyp.resolvent_path+"/3d/"
dirCreator(dir)

Ss_ref = [0,1]
ms_ref = [-2,0,2]
Sts_ref = [.025,.5]

def frameArgs(duration):
    return {"frame": {"duration": duration}, "mode": "immediate", "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},}

# Nozzle surface
y = np.cos(np.linspace(0,np.pi,15))
x,y = np.meshgrid([0,1],y)
z = np.sqrt(1-y**2)
up_nozzle   = go.Surface(x=x, y=y, z= z, colorscale=[[0,'black'],[1,'black']], opacity=.5, showscale=False, name="nozzle")
down_nozzle = go.Surface(x=x, y=y, z=-z, colorscale=[[0,'black'],[1,'black']], opacity=.5, showscale=False, name="nozzle")

# Custom nonuniform grids
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

"""
if p0:
	fig = go.Figure(data=[go.Scatter3d(x=XYZf_sk[0],y=XYZf_sk[1],z=XYZf_sk[2], mode='markers', marker={"size":3,"opacity":.4})])
	fig.write_html("XYZf_sk.html")
	fig = go.Figure(data=[go.Scatter3d(x=XYZr_sk[0],y=XYZr_sk[1],z=XYZr_sk[2], mode='markers', marker={"size":3,"opacity":.4})])
	fig.write_html("XYZr_sk.html")
	fig = go.Figure(data=[go.Scatter3d(x=XYZf_kh[0],y=XYZf_kh[1],z=XYZf_kh[2], mode='markers', marker={"size":3,"opacity":.4})])
	fig.write_html("XYZf_kh.html")
	fig = go.Figure(data=[go.Scatter3d(x=XYZr_kh[0],y=XYZr_kh[1],z=XYZr_kh[2], mode='markers', marker={"size":3,"opacity":.4})])
	fig.write_html("XYZr_kh.html")

for S in Ss_ref:
	for m in ms_ref:
		for St in Sts_ref:
			file_name=dir+f"Re={Re:d}_S={S:.1f}_m={m:d}_St={St:.2f}".replace('.',',')+".html" # Usual Re & St based on D but who cares
			if p0: print(f"Currently beautifying (Re,S,m,St)=({Re},{S:.1f},{m},{St:.2f})",flush=True)
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
				#isos_f=spyp.computeIsosurfaces(m,XYZf_sk,.1,spyp.readMode("forcing", Re,S,m,St),2,'Earth', "axial forcing")
				isos_r=spyp.computeIsosurfaces(m,XYZr_st,.1,spyp.readMode("response",Re,S,m,St),2,'Picnic',"axial response")
			elif St == Sts_ref[0] and m < 0:
				#isos_f=spyp.computeIsosurfaces(m,XYZf_sk,.1,spyp.readMode("forcing", Re,S,m,St),2,'Earth', "axial forcing")
				isos_r=spyp.computeIsosurfaces(m,XYZr_sw,.1,spyp.readMode("response",Re,S,m,St),2,'Picnic',"axial response")
			elif St == Sts_ref[-1]:
				#isos_f=spyp.computeIsosurfaces(m,XYZf_kh,.1,spyp.readMode("forcing", Re,S,m,St),2,'Earth', "axial forcing")
				isos_r=spyp.computeIsosurfaces(m,XYZr_kh,.1,spyp.readMode("response",Re,S,m,St),2,'Picnic',"axial response")
			else:
				#isos_f=spyp.computeIsosurfaces(m,XYZf_cr,.1,spyp.readMode("forcing", Re,S,m,St),2,'Earth', "axial forcing")
				isos_r=spyp.computeIsosurfaces(m,XYZr_nr,.1,spyp.readMode("response",Re,S,m,St),2,'Picnic',"axial response")
			# Animation
			if p0:
				fig = go.Figure(data=[#isos_f[0],
									isos_r[0], up_nozzle, down_nozzle],
								layout=go.Layout(updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])])]),
								frames=[go.Frame(data=[#isos_f[i],
								isos_r[i], up_nozzle, down_nozzle], name=str(i)) for i in range(len(isos_r))])
				fig.update_layout(sliders=[{"pad": {"b": 10, "t": 60}, "len": .9, "x": .1, "y": 0,
											"steps": [{"args": [[f.name], frameArgs(0)], "label": str(k),
													"method": "animate",} for k, f in enumerate(fig.frames)],}]	)
				fig.update_coloraxes(showscale=False)
				fig.update_layout(scene_aspectmode='data')
				fig.write_html(file_name)"""