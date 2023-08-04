# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from sys import path
from os.path import isfile

path.append('/home/shared/cases/nozzle/src/')

from setup import *
from grids_3d import *
from spyp import SPYP # Must be after setup
from helpers import dirCreator

spyp = SPYP(params,data_path,pert_mesh,direction_map)

# Actual plotting
dir=spyp.resolvent_path+"/3d/"
dirCreator(dir)

if p0:
	for mesh in []:
		exec("fig = go.Figure(data=[go.Scatter3d(x="+mesh+"[0],y="+mesh+"[1],z="+mesh+"[2], mode='markers', marker={'size':3,'opacity':.4})]); fig.write_html('"+dir+mesh+".html')")

directions=list(direction_map.keys())

print_list=[{'S':1,'m':-2,'St':7.3057e-03/2,'XYZ':[XYZr_esn,XYZf_es,XYZc_es],'print_f':True,'print_U':False,'all_dirs':False},
			{'S':1,'m': 2,'St':7.3057e-03/2,'XYZ':[XYZr_esp,XYZf_es,XYZc_es],'print_f':True,'print_U':False,'all_dirs':False},
			#{'S':1,'m': 2,'St':0,		    'XYZ':[XYZr_st,XYZf_cr,XYZc_st],'print_f':True,'print_U':True, 'all_dirs':False},
			#{'S':1,'m':-2,'St':0,		    'XYZ':[XYZr_sw,XYZf_cr,XYZc_sw],'print_f':True,'print_U':True, 'all_dirs':False},
			#{'S':1,'m': 0,'St':.5,		    'XYZ':[XYZr_kh,XYZf_kh],		'print_f':True,'print_U':False,'all_dirs':False},
			#{'S':1,'m': 0,'St':0,		    'XYZ':[XYZr_esp,XYZf_es],		'print_f':True,'print_U':False,'all_dirs':False},
			#{'S':0,'m': 0,'St':.5,		    'XYZ':[XYZr_kh,XYZf_kh],		'print_f':True,'print_U':False,'all_dirs':False},
			#{'S':0,'m': 0,'St':0,		    'XYZ':[XYZr_cl,XYZf_cl],		'print_f':True,'print_U':False,'all_dirs':False},
			#{'S':0,'m':-2,'St':0,		    'XYZ':[XYZr_cl,XYZf_cl],		'print_f':True,'print_U':False,'all_dirs':False},
			#{'S':0,'m':-2,'St':7.3057e-03/2,'XYZ':[XYZr_ln,XYZf_es],		'print_f':True,'print_U':False,'all_dirs':False},
			#{'S':0,'m': 2,'St':0,		    'XYZ':[XYZr_cl,XYZf_cl],		'print_f':False,'print_U':False,'all_dirs':False}
]
S_save=-1

for dat in print_list:
	S,m,St,XYZ,all_dirs=dat['S'],dat['m'],dat['St'],dat['XYZ'],dat['all_dirs']
	print_f,print_U=dat['print_f'],dat['print_U']
	if S_save!=S: # If current swirl isn't correct, load the correct one
		spyb.loadBaseflow(Re,S,False)
		S_save=S
	small_dat={'Re':Re,'S':S,'m':m,'St':St}
	if p0: print(f"Currently beautifying (Re,S,m,St)=({Re},{S:.1f},{m},{St:.2f})",flush=True)
	file_name=dir+f"Re={Re:d}_S={S:.1f}_m={m:d}_St={St:.4e}_f={print_f:d}_U={print_U:d}".replace('.',',') # Usual Re & St based on D
	"""if isfile(file_name+"_dir=x.html"):
		if all_dirs:
			if isfile(file_name+"_dir=r.html") and isfile(file_name+"_dir=theta.html"):
				if p0: print("Found html files, moving on...",flush=True)
				continue
		else:
			if p0: print("Found html file, moving on...",flush=True)
			continue"""
	isos_r=spyp.computeIsosurfaces("response",small_dat,XYZ[0],.1,1,'Picnic',"response",all_dirs)
	if print_f: isos_f=spyp.computeIsosurfaces("forcing",small_dat,XYZ[1],.1,1,'Earth', "forcing",all_dirs)
	if print_U: quiv_U=spyb.computeQuiver(XYZ[2],"Greens")
	# Plus d'animation - on économise la place !
	if p0:
		data=[isos_r[0][0], up_nozzle, down_nozzle, dot]
		if print_f: data.append(isos_f[0][0])
		if print_U: data.append(quiv_U)
		fig = go.Figure(data=data)
		fig.update_coloraxes(showscale=False)
		fig.update_layout(scene_aspectmode='data',scene_camera={"up":{'x':0,'y':0,'z':1},"center":{'x':0,'y':0,'z':-.25},"eye":{'x':-1.5,'y':-1.25,'z':1.25}})
		fig.write_html(f"{file_name}_dir={directions[0]}.html")
		print(f"Wrote {file_name}_dir={directions[0]}.html",flush=True)

		if all_dirs:
			for i in (1,2):
				data=[isos_r[i][0], up_nozzle, down_nozzle]
				if print_f: data.append(isos_f[i][0])
				if print_U: data.append(quiv_U)
				fig = go.Figure(data=data)
				fig.update_coloraxes(showscale=False)
				fig.update_layout(scene_aspectmode='data',scene_camera={"up":{'x':0,'y':0,'z':1},"center":{'x':0,'y':0,'z':-.25},"eye":{'x':-1.5,'y':-1.25,'z':1.25}})
				fig.write_html(f"{file_name}_dir={directions[i]}.html")
				print(f"Wrote {file_name}_dir={directions[i]}.html",flush=True)