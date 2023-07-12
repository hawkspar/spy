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
dir=spyp.resolvent_path+"3d/"
dirCreator(dir)

directions=list(direction_map.keys())

print_list=[{'S':1,'m':-2,'St':7.3057e-03/2,'XYZ':[XYZt_es,XYZf_es,XYZb],'print_f':False,'print_U':True}#,
			#{'S':1,'m': 2,'St':7.3057e-03/2,'XYZ':[XYZr_es,XYZf_es,XYZc_es],'print_f':True,'print_U':True},
			#{'S':1,'m': 2,'St':0,		    'XYZ':[XYZr_st,XYZf_cr,XYZc_st],'print_f':True,'print_U':True},
			#{'S':1,'m':-2,'St':0,		    'XYZ':[XYZr_sw,XYZf_cr,XYZc_sw],'print_f':True,'print_U':True},
			#{'S':1,'m': 0,'St':.5,		    'XYZ':[XYZr_kh,XYZf_kh],		'print_f':True,'print_U':False},
			#{'S':0,'m': 0,'St':.5,		    'XYZ':[XYZr_kh,XYZf_kh],		'print_f':True,'print_U':False},
			#{'S':0,'m': 0,'St':0,		    'XYZ':[XYZr_cl,XYZf_cl],			'print_f':True,'print_U':False},
			#{'S':0,'m':-2,'St':0,		    'XYZ':[XYZr_cl,XYZf_cl],		'print_f':True,'print_U':False},
			#{'S':0,'m': 2,'St':0,		    'XYZ':[XYZr_cl,XYZf_cl],		'print_f':False,'print_U':False}
]
S_save=-1

for dat in print_list:
	S,m,St,XYZ=dat['S'],dat['m'],dat['St'],dat['XYZ']
	print_f,print_U=dat['print_f'],dat['print_U']
	if S_save!=S:
		spyb.loadBaseflow(Re,S,False)
		S_save=S
	small_dat={'Re':Re,'S':S,'m':m,'St':St}
	if p0: print(f"Currently beautifying (Re,S,m,St)=({Re},{S:.1f},{m},{St:.2f})",flush=True)
	file_name=dir+f"rot_Re={Re:d}_S={S:.1f}_m={m:d}_St={St:.4e}_f={print_f:d}_U={print_U:d}".replace('.',',') # Usual Re & St based on D
	"""if isfile(file_name+"_dir=x.html"):
		if all_dirs:
			if isfile(file_name+"_dir=r.html") and isfile(file_name+"_dir=theta.html"):
				if p0: print("Found html files, moving on...",flush=True)
				continue
		else:
			if p0: print("Found html file, moving on...",flush=True)
			continue"""
	cones_r=spyp.compute3DCurlsCones("response",small_dat,XYZ[0],1,1,'Picnic',"response")
	if print_f: cones_f=spyp.compute3DCurlsCones("forcing",small_dat,XYZ[1],1,1,'Earth', "forcing")
	if print_U: isos_U=spyb.computeIsosurfaces(XYZ[2],.9,"Greens",True)
	# Plus d'animation - on économise la place !
	if p0:
		for i,iso_U in enumerate(isos_U):
			data=[cones_r[0], up_nozzle, down_nozzle]
			if print_f: data.append(cones_f[0])
			if print_U: data.append(iso_U)
			fig = go.Figure(data=data)
			fig.update_coloraxes(showscale=False)
			fig.update_layout(scene_aspectmode='data')
			fig.write_html(f"{file_name}_dir={list(direction_map.keys())[i]}.html")
			print(f"Wrote {file_name}_dir={list(direction_map.keys())[i]}.html",flush=True)